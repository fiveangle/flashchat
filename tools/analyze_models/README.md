# Hugging Face Model Architecture & GPU Bandwidth Analyzer

This tool reads a list of Hugging Face model URLs, fetches their `config.json`, and calculates detailed architectural parameters and per-token memory bandwidth requirements. It is specifically designed for DevOps and SREs architecting GPU server farms.

## Features
- **The "Memory Wall" Analysis:** Calculates exact VRAM bandwidth required per token for Native (FP16), Q8, and Q4 quantizations.
- **Custom Attention Math:** Identifies and calculates exact parameter counts for custom mechanisms like DeepSeek's Multi-head Latent Attention (MLA) and MiniMax's Sparse Attention (MSA).
- **Shared Experts Integration:** Detects "always-active" shared experts and includes them in the active per-token bandwidth calculations.
- **Advanced Visualizations:** Generates three technical charts directly in the Excel output:
  1. **Memory Bandwidth per Token:** The primary bottleneck for autoregressive generation throughput.
  2. **Total VRAM vs. Active Compute:** Highlights the MoE "Sparsity Tax" (VRAM footprint vs. actual compute cost).
  3. **Active Parameter Composition:** Shows exactly where the per-token bandwidth is being spent (Attention vs. MLP vs. Overhead).

## Prerequisites

You need [uv](https://docs.astral.sh/uv/), an extremely fast Python package and project manager.

**Install uv:**
- **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## How to Run

The script carries its dependencies as [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so `uv` provisions an isolated environment automatically — there is no `pyproject.toml` to install and no venv to manage.

1. List the models to analyze, one Hugging Face URL per line, in a `models.txt` file in this directory (a sample `models.txt` is included; the script also accepts a file named `models`).
2. Run the script from this directory:
   ```bash
   uv run analyze_models.py
   ```
3. Results are written to `model_architecture_analysis.xlsx` (data tables plus the three charts described above) in the same directory.