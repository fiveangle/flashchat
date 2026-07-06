# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "pandas",
#     "openpyxl",
#     "xlsxwriter",
# ]
# ///

import requests
import pandas as pd
import sys
import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

def get_config_val(config: dict, *keys, default=None):
    """Safely extract a value from a config dictionary using multiple fallback keys."""
    for key in keys:
        if key in config:
            return config[key]
    return default

def fetch_model_config(repo_id: str) -> dict:
    """Fetch the config.json using the 'hf' CLI if available, else fallback to HTTP."""
    
    # 1. Try using the 'hf' CLI tool if it's installed on the system
    if shutil.which('hf'):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Execute: hf download <repo_id> config.json --local-dir <temp_dir> --quiet
                cmd = ['hf', 'download', repo_id, 'config.json', '--local-dir', temp_dir, '--quiet']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    config_path = Path(temp_dir) / 'config.json'
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
            except Exception as e:
                print(f"    (Note: 'hf' CLI failed for {repo_id}, falling back to HTTP: {e})")

    # 2. Fallback to standard HTTP request if 'hf' isn't installed or failed
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching config for {repo_id}: {e}")
        return None

def calculate_architecture_and_bandwidth(repo_id: str, config: dict) -> dict:
    """Calculate parameter counts and memory bandwidth based on config.json."""
    
    # 1. Extract core architecture dimensions
    h = get_config_val(config, 'hidden_size', 'n_embd', 'd_model', default=0)
    L = get_config_val(config, 'num_hidden_layers', 'n_layer', 'num_layers', default=0)
    a = get_config_val(config, 'num_attention_heads', 'n_head', 'num_heads', default=1)
    kv = get_config_val(config, 'num_key_value_heads', 'num_kv_heads', 'multi_query_group_num', default=a)
    i = get_config_val(config, 'intermediate_size', 'ffn_hidden_size', default=0)
    v = get_config_val(config, 'vocab_size', default=0)
    
    # MoE specific parameters
    e = get_config_val(config, 'num_local_experts', 'num_experts', 'n_routed_experts', default=1)
    e_active = get_config_val(config, 'num_experts_per_tok', 'num_selected_experts', default=1)
    
    # Shared / Always-Active Experts
    shared_experts = get_config_val(config, 'n_shared_experts', 'num_shared_experts', default=0)
    shared_intermediate = get_config_val(config, 'shared_expert_intermediate_size', 'shared_intermediate_size', default=i)

    is_moe = e > 1

    # 2. Determine Attention Type based on model architecture
    model_type = config.get('model_type', '').lower()
    arch = config.get('architectures', [''])[0].lower()

    attn_type = "Standard (GQA/MQA)"
    if 'deepseek' in model_type or 'deepseek' in arch:
        attn_type = "Custom (MLA)"
    elif 'minimax' in model_type or 'minimax' in arch:
        attn_type = "Custom (MSA)"
    elif 'glm' in model_type or 'chatglm' in arch:
        attn_type = "Custom (GLM/DSA)"
    elif 'kimi' in model_type or 'moonshot' in arch:
        attn_type = "Custom (Kimi MoE)"

    # 3. Calculate Parameter Counts
    emb_params = v * h
    lm_head_params = 0 if config.get('tie_word_embeddings', False) else (v * h) 
    
    # Attention per layer
    if attn_type == "Custom (MLA)":
        q_lora = get_config_val(config, 'q_lora_rank', default=0)
        kv_lora = get_config_val(config, 'kv_lora_rank', default=0)
        qk_nope = get_config_val(config, 'qk_nope_head_dim', default=0)
        qk_rope = get_config_val(config, 'qk_rope_head_dim', default=0)
        v_head = get_config_val(config, 'v_head_dim', default=0)

        if q_lora and kv_lora and qk_nope:
            q_dim = a * (qk_nope + qk_rope)
            kv_dim = a * (qk_nope + v_head)
            q_a = h * q_lora
            q_b = q_lora * q_dim
            kv_a = h * (kv_lora + qk_rope)
            kv_b = kv_lora * kv_dim
            o_proj = (a * v_head) * h
            attn_per_layer = q_a + q_b + kv_a + kv_b + o_proj
        else:
            attn_per_layer = (h ** 2) * (2 + 2 * (kv / a))
    else:
        attn_per_layer = (h ** 2) * (2 + 2 * (kv / a))
    
    # MLP per layer
    if is_moe:
        total_mlp_per_layer = e * 3 * h * i
        active_mlp_per_layer = e_active * 3 * h * i
        router_params = h * e
    else:
        total_mlp_per_layer = 3 * h * i
        active_mlp_per_layer = total_mlp_per_layer
        router_params = 0

    # Shared Experts (Always Active)
    shared_params_per_layer = 0
    if shared_experts > 0:
        shared_params_per_layer = shared_experts * 3 * h * shared_intermediate

    ln_per_layer = 2 * h
    final_norm = h

    # --- Breakdown Calculations for Charting ---
    emb_total = emb_params + lm_head_params
    attn_total = L * attn_per_layer
    mlp_active = L * (active_mlp_per_layer + shared_params_per_layer)
    overhead_active = L * (router_params + ln_per_layer) + final_norm
    inactive_moe = L * (total_mlp_per_layer - active_mlp_per_layer) if is_moe else 0

    total_params = emb_total + attn_total + L * total_mlp_per_layer + L * router_params + L * ln_per_layer + final_norm
    active_params = emb_total + attn_total + mlp_active + overhead_active

    # 4. Calculate Memory Bandwidth per Token
    BYTES_PER_GIB = 1024 ** 3 
    
    bw_native = (active_params * 2) / BYTES_PER_GIB
    bw_q8 = (active_params * 1) / BYTES_PER_GIB
    bw_q4 = (active_params * 0.5) / BYTES_PER_GIB

    return {
        "Model": repo_id,
        "Attention Type": attn_type,
        "Hidden Size": h,
        "Layers": L,
        "Attn Heads": a,
        "KV Heads": kv,
        "Intermediate Size": i,
        "Vocab Size": v,
        "Total Experts": e if is_moe else "Dense",
        "Active Experts": e_active if is_moe else "Dense",
        "Shared Experts": shared_experts if shared_experts > 0 else "None",
        "Total Params (B)": round(total_params / 1e9, 2),
        "Active Params (B)": round(active_params / 1e9, 2),
        "Active: Embed (B)": round(emb_total / 1e9, 2),
        "Active: Attn (B)": round(attn_total / 1e9, 2),
        "Active: MLP (B)": round(mlp_active / 1e9, 2),
        "Active: Overhead (B)": round(overhead_active / 1e9, 2),
        "Inactive MoE (B)": round(inactive_moe / 1e9, 2),
        "BW Native (GiB/tok)": round(bw_native, 3),
        "BW Q8 (GiB/tok)": round(bw_q8, 3),
        "BW Q4 (GiB/tok)": round(bw_q4, 3),
    }

def format_excel_and_charts(writer, df):
    """Apply professional formatting and generate DevOps-focused charts."""
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # num_rows is the count of data rows. 
    # Since row 0 is headers, our data ends at row index = len(df).
    last_row = len(df) 
    
    # Formats
    header_fmt = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#2F5496', 'font_color': 'white', 'border': 1})
    num_fmt = workbook.add_format({'num_format': '#,##0.00'})
    bw_fmt = workbook.add_format({'num_format': '0.000'})
    text_fmt = workbook.add_format()

    # Write headers
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_fmt)

    # Column widths
    worksheet.set_column('A:A', 35, text_fmt) # Model
    worksheet.set_column('B:B', 20, text_fmt) # Attention Type
    worksheet.set_column('C:J', 12, num_fmt)  # Architecture details
    worksheet.set_column('K:M', 14, text_fmt) # Experts
    worksheet.set_column('N:R', 16, num_fmt)  # Params Breakdown
    worksheet.set_column('S:U', 18, bw_fmt)   # Bandwidth
    
    worksheet.freeze_panes(1, 1) 

    # ==========================================
    # CHART 1: Memory Bandwidth per Token
    # ==========================================
    chart1 = workbook.add_chart({'type': 'column', 'subtype': 'clustered'})
    chart1.set_title({'name': 'Memory Bandwidth per Token (GiB) - The Autoregressive Bottleneck'})
    chart1.set_y_axis({'name': 'GiB per Token', 'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}}})
    chart1.set_x_axis({'name': 'Model', 'label_position': 'low', 'num_font': {'rotation': -45, 'size': 9}})
    chart1.set_size({'width': 720, 'height': 450})
    chart1.set_legend({'position': 'bottom'})

    # Columns S(18), T(19), U(20)
    for col_idx, name, color in [(18, 'Native (FP16)', '#C00000'), (19, 'Q8', '#ED7D31'), (20, 'Q4', '#70AD47')]:
        chart1.add_series({
            'name': name,
            'categories': ['Sheet1', 1, 0, last_row, 0],  
            'values': ['Sheet1', 1, col_idx, last_row, col_idx], 
            'fill': {'color': color},
            'gap': 50,
        })
    worksheet.insert_chart('W2', chart1)

    # ==========================================
    # CHART 2: Total VRAM vs Active Compute
    # ==========================================
    chart2 = workbook.add_chart({'type': 'bar', 'subtype': 'clustered'})
    chart2.set_title({'name': 'Total VRAM Footprint vs. Active Compute Cost (Billions of Params)'})
    chart2.set_x_axis({'name': 'Parameters (B)', 'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}}})
    chart2.set_y_axis({'name': 'Model', 'reverse': True, 'num_font': {'size': 9}})
    chart2.set_size({'width': 720, 'height': 450})
    chart2.set_legend({'position': 'bottom'})

    # Columns L(11), M(12)
    chart2.add_series({
        'name': 'Total Params (VRAM Capacity Required)', 
        'categories': ['Sheet1', 1, 0, last_row, 0], 
        'values': ['Sheet1', 1, 11, last_row, 11], 
        'fill': {'color': '#4472C4'}
    })
    chart2.add_series({
        'name': 'Active Params (Compute/Bandwidth Cost)', 
        'categories': ['Sheet1', 1, 0, last_row, 0], 
        'values': ['Sheet1', 1, 12, last_row, 12], 
        'fill': {'color': '#FFC000'}
    })
    worksheet.insert_chart('W20', chart2)

    # ==========================================
    # CHART 3: Active Parameter Composition
    # ==========================================
    chart3 = workbook.add_chart({'type': 'column', 'subtype': 'percent_stacked'})
    chart3.set_title({'name': 'Active Parameter Composition (100% of Per-Token Bandwidth)'})
    chart3.set_y_axis({'name': '% of Active Params', 'num_format': '0%'})
    chart3.set_x_axis({'label_position': 'low', 'num_font': {'rotation': -45, 'size': 9}})
    chart3.set_size({'width': 720, 'height': 450})
    chart3.set_legend({'position': 'bottom'})

    # Columns N(13), O(14), P(15), Q(16)
    colors = ['#5B9BD5', '#70AD47', '#FFC000', '#9E480E']
    names = ['Embeddings & LM Head', 'Attention (incl. MLA/MSA)', 'Active MLP (Routed + Shared)', 'Overhead (Routers & LayerNorms)']
    
    for idx, (col_idx, name, color) in enumerate(zip([13, 14, 15, 16], names, colors)):
        chart3.add_series({
            'name': name,
            'categories': ['Sheet1', 1, 0, last_row, 0],
            'values': ['Sheet1', 1, col_idx, last_row, col_idx],
            'fill': {'color': color},
        })
    worksheet.insert_chart('W38', chart3)

def main():
    # Check for either 'models' or 'models.txt'
    if Path("models").exists():
        input_file = Path("models")
    elif Path("models.txt").exists():
        input_file = Path("models.txt")
    else:
        print("Error: Neither 'models' nor 'models.txt' found. Please create one of these files with one HF URL per line.")
        sys.exit(1)

    output_file = Path("model_architecture_analysis.xlsx")

    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("The input file is empty.")
        sys.exit(1)

    results = []
    print(f"Processing {len(urls)} models...")

    for idx, url in enumerate(urls):
        repo_id = url.replace("https://huggingface.co/", "").strip('/')
        print(f" -> Analyzing {repo_id}...")
        
        config = fetch_model_config(repo_id)
        if config:
            try:
                data = calculate_architecture_and_bandwidth(repo_id, config)
                results.append(data)
            except Exception as e:
                print(f"    ! Calculation error for {repo_id}: {e}")
                results.append({"Model": repo_id, "Error": str(e)})
        else:
            results.append({"Model": repo_id, "Error": "Failed to fetch config.json"})

        # Pause for 1.5 seconds between connections to respect Hugging Face rate limits
        if idx < len(urls) - 1:
            time.sleep(1.5)

    df = pd.DataFrame(results)
    
    # Ensure Bandwidth columns are at the far right
    bw_cols = [c for c in df.columns if c.startswith("BW")]
    other_cols = [c for c in df.columns if not c.startswith("BW")]
    df = df[other_cols + bw_cols]

    print(f"\nSaving results and generating charts to {output_file}...")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        format_excel_and_charts(writer, df)

    print("Done! Open the Excel file to see the data and technical charts.")

if __name__ == "__main__":
    main()