# Makefile for Flashchat — Pure C/Metal MoE inference engine
#
# Targets:
#   make           — build inference binaries
#   make run       — single expert forward pass
#   make verify    — verify Metal vs CPU reference
#   make bench     — benchmark single expert (10 iterations)
#   make moe       — full MoE forward pass (K experts, single layer)
#   make moebench — benchmark MoE (10 iterations)
#   make full      — full model MoE forward pass (K=4)
#   make fullbench — benchmark full model forward (3 iterations)
#   make chat      — build interactive chat TUI
#   make api-smoke — run HTTP API smoke test
#   make cli-smoke — run Flashchat CLI smoke test
#   make manage-smoke — run model management storage smoke test
#   make tool-template-smoke — run native tool template render/parser smoke test
#   make quant-helper-smoke — run native checkpoint quantization helper tests
#   make tokenizer-export-smoke — run tokenizer export helper tests
#   make native-qwen-compile-smoke — run native Qwen BF16 compiler smoke test
#   make mpp-tensorops-smoke — run Metal 4 MPP TensorOps compile smoke test
#   make mpp-tensorops-runtime-smoke — run Metal 4 MPP TensorOps runtime smoke test
#   make mpp-tensorops-bench — benchmark TensorOps affine matmul against current v5
#   make mtp-config-smoke — run MTP config/profile precedence smoke test
#   make model-add-config-smoke — run add-model configuration smoke test
#   make model-edit-config-smoke — run edit-model registry smoke test
#   make test      — run all functional smoke tests
#   make help      — list available targets
#   make clean     — remove build artifacts
#   make archive-debug — archive repo-local debug contents under debug/.archived
#   make clean-venv — remove Python setup virtual environment
#   make distclean — remove build artifacts, repo-local debug, and setup venv
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

SHELL := /bin/bash

BUILD_DIR = metal_infer

ifeq ($(origin CC),default)
CC = clang
endif
OPT ?= aggressive

FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
BASE_CFLAGS = -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
BASE_LDFLAGS = -lpthread -lcompression

VALID_OPTS = aggressive conservative debug
ifeq ($(filter $(OPT),$(VALID_OPTS)),)
$(error Unknown OPT='$(OPT)'. Use one of: $(VALID_OPTS))
endif

cc-option = $(strip $(shell tmp=$$(mktemp /tmp/flashchat-cc-option.XXXXXX); \
	printf 'int main(void){return 0;}\n' | $(CC) -x objective-c $(BASE_CFLAGS) $(1) -c -o "$$tmp" - >/dev/null 2>&1 && printf '%s' '$(1)'; \
	rm -f "$$tmp"))

link-option = $(strip $(shell tmp=$$(mktemp /tmp/flashchat-link-option.XXXXXX); rm -f "$$tmp"; \
	printf 'int main(void){return 0;}\n' | $(CC) -x objective-c $(BASE_CFLAGS) $(BASE_LDFLAGS) $(1) -o "$$tmp" - >/dev/null 2>&1 && printf '%s' '$(1)'; \
	rm -f "$$tmp"))

APPLE_CPU_NAME = $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed -n 's/^apple \([a-z0-9]*\).*/\1/p')
DETECTED_MCPU_FLAG = $(if $(APPLE_CPU_NAME),-mcpu=apple-$(APPLE_CPU_NAME))
CPU_CFLAGS := $(call cc-option,-mcpu=native)
ifeq ($(CPU_CFLAGS),)
CPU_CFLAGS := $(call cc-option,$(DETECTED_MCPU_FLAG))
endif

AGGRESSIVE_LTO = $(call link-option,-flto)
AGGRESSIVE_EXTRA_CFLAGS = \
	$(call cc-option,-ffast-math) \
	$(call cc-option,-funroll-loops) \
	$(call cc-option,-fvectorize) \
	$(call cc-option,-fslp-vectorize) \
	$(call cc-option,-ftree-vectorize) \
	$(call cc-option,-falign-functions=16)

ifeq ($(OPT),aggressive)
OPT_CFLAGS = -O3 $(CPU_CFLAGS) $(AGGRESSIVE_LTO) $(AGGRESSIVE_EXTRA_CFLAGS)
OPT_LDFLAGS = $(AGGRESSIVE_LTO)
endif
ifeq ($(OPT),conservative)
OPT_CFLAGS = -O2 $(CPU_CFLAGS)
OPT_LDFLAGS =
endif
ifeq ($(OPT),debug)
OPT_CFLAGS = -O0 -g $(CPU_CFLAGS)
OPT_LDFLAGS =
endif

CFLAGS = $(BASE_CFLAGS) $(OPT_CFLAGS)
LDFLAGS = $(BASE_LDFLAGS) $(OPT_LDFLAGS)
CHAT_CFLAGS = $(if $(filter debug,$(OPT)),-O0 -g,-O2 $(CPU_CFLAGS)) -Wall -fobjc-arc

TARGET = $(BUILD_DIR)/metal_infer
MAIN_SRC = $(BUILD_DIR)/main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = $(BUILD_DIR)/shaders.metal
SHADER_AIR = $(BUILD_DIR)/shaders.air
SHADER_LIB = $(BUILD_DIR)/shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = $(BUILD_DIR)/infer
INFER_SRC = $(BUILD_DIR)/infer.m

# Chat TUI (interactive multi-turn)
CHAT_TARGET = $(BUILD_DIR)/chat
CHAT_SRC = $(BUILD_DIR)/chat.m
LINENOISE_SRC = $(BUILD_DIR)/linenoise.c
LINENOISE_HDR = $(BUILD_DIR)/linenoise.h

.PHONY: all clean archive-debug clean-venv distclean help print-build-config run verify bench moe moebench full fullbench fast metallib metal_infer infer chat build-infer infer-run chat-run build-chat api-smoke cli-smoke manage-smoke chat-render-smoke tool-template-smoke cache-roundtrip-smoke quant-helper-smoke tokenizer-export-smoke native-qwen-compile-smoke mpp-tensorops-smoke mpp-tensorops-runtime-smoke mpp-tensorops-bench mtp-config-smoke model-add-config-smoke model-edit-config-smoke test bench-api bench-report

define RUN_ENGINE_BENCH
	@bash -c 'set -eo pipefail; \
	source ./lib/config.sh; \
	flashchat_load_config; \
	export FLASHCHAT_MODEL="$$(flashchat_get MODEL)"; \
	export FLASHCHAT_MODEL_PATH="$$(flashchat_get MODEL_PATH)"; \
	export FLASHCHAT_MODEL_CONFIG="$$(flashchat_get MODEL_CONFIG)"; \
	if [[ -z "$$FLASHCHAT_MODEL_PATH" || "$$FLASHCHAT_MODEL_PATH" == *"<snapshot>"* || ! -d "$$FLASHCHAT_MODEL_PATH" ]]; then \
		echo "ERROR: Model is not downloaded for $$FLASHCHAT_MODEL."; \
		echo "Expected model snapshot: $$FLASHCHAT_MODEL_PATH"; \
		echo "Run ./flashchat setup first, or select a configured model with downloaded weights."; \
		exit 1; \
	fi; \
	packed_dir="$$FLASHCHAT_MODEL_PATH/flashchat/packed_experts"; \
	if [[ ! -f "$$packed_dir/layer_00.bin" ]]; then \
		echo "ERROR: Engine benchmark artifacts are not available for $$FLASHCHAT_MODEL."; \
		echo "Expected: $$packed_dir/layer_00.bin"; \
		echo "Run ./flashchat setup first, or select a configured model with packed experts."; \
		exit 1; \
	fi; \
	echo "Using Flashchat model $$FLASHCHAT_MODEL"; \
	cd $(BUILD_DIR); \
	./metal_infer --model-id "$$FLASHCHAT_MODEL" --model "$$FLASHCHAT_MODEL_PATH" $(1)'
endef

all: $(TARGET) $(INFER_TARGET)

help:
	@printf "Flashchat make targets\n"
	@printf "\n"
	@printf "Build:\n"
	@printf "  make               Build main benchmark and inference binaries\n"
	@printf "  make all           Build main benchmark and inference binaries\n"
	@printf "  make metal_infer   Build main benchmark binary\n"
	@printf "  make infer         Build inference server/engine\n"
	@printf "  make build-infer   Alias for infer\n"
	@printf "  make chat          Build interactive chat client\n"
	@printf "  make build-chat    Alias for chat\n"
	@printf "  make metallib      Precompile Metal shaders\n"
	@printf "  make print-build-config  Show compiler and optimization settings\n"
	@printf "\n"
	@printf "Build options:\n"
	@printf "  OPT=aggressive     Fastest probed local build (default)\n"
	@printf "  OPT=conservative   Native CPU, fewer risky optimization flags\n"
	@printf "  OPT=debug          Debug symbols, no speed-oriented flags\n"
	@printf "  CC=clang           Override compiler command\n"
	@printf "\n"
	@printf "Run:\n"
	@printf "  make infer-run     Run a short inference prompt\n"
	@printf "  make chat-run      Launch the chat client\n"
	@printf "\n"
	@printf "Benchmarks:\n"
	@printf "  make run           Single expert forward pass\n"
	@printf "  make verify        Metal vs CPU reference verification\n"
	@printf "  make fast          Fast path verification\n"
	@printf "  make bench         Single expert benchmark\n"
	@printf "  make moe           MoE forward pass\n"
	@printf "  make moebench      MoE benchmark\n"
	@printf "  make full          Full model forward pass\n"
	@printf "  make fullbench     Full model benchmark\n"
	@printf "\n"
	@printf "Tests:\n"
	@printf "  make cli-smoke     Run Flashchat CLI smoke test\n"
	@printf "  make manage-smoke  Run model management storage smoke test\n"
	@printf "  make chat-render-smoke  Run chat TUI render smoke test\n"
	@printf "  make tool-template-smoke  Run native tool template render/parser smoke test\n"
	@printf "  make cache-roundtrip-smoke  Run disk-cache save/load roundtrip self-test\n"
	@printf "  make quant-helper-smoke  Run native checkpoint quantization helper tests\n"
	@printf "  make tokenizer-export-smoke  Run tokenizer export helper tests\n"
	@printf "  make native-qwen-compile-smoke  Run native Qwen BF16 compiler smoke test\n"
	@printf "  make mpp-tensorops-smoke  Run Metal 4 MPP TensorOps compile smoke test\n"
	@printf "  make mpp-tensorops-runtime-smoke  Run Metal 4 MPP TensorOps runtime smoke test\n"
	@printf "  make mpp-tensorops-bench  Benchmark TensorOps affine matmul against current v5\n"
	@printf "  make mtp-config-smoke  Run MTP config/profile precedence smoke test\n"
	@printf "  make model-add-config-smoke  Run add-model configuration smoke test\n"
	@printf "  make model-edit-config-smoke  Run edit-model registry smoke test\n"
	@printf "  make api-smoke     Run HTTP API smoke test\n"
	@printf "  make test          Run all functional smoke tests\n"
	@printf "  make bench-api     Run API performance regression benchmark (per registry model)\n"
	@printf "  make bench-report  Compare latest benchmark vs prior commits, flag regressions\n"
	@printf "\n"
	@printf "Maintenance:\n"
	@printf "  make clean         Remove build artifacts and archive repo-local ./debug contents\n"
	@printf "  make archive-debug Archive repo-local ./debug contents under debug/.archived\n"
	@printf "  make clean-venv    Remove Python setup virtual environment\n"
	@printf "  make distclean     Remove build artifacts, repo-local ./debug, and setup venv\n"

print-build-config:
	@printf "Compiler command: %s\n" "$(CC)"
	@printf "Compiler path: "
	@command -v $(firstword $(CC)) 2>/dev/null || printf "%s\n" "$(firstword $(CC))"
	@$(CC) --version | head -1
	@printf "Optimization profile: %s\n" "$(OPT)"
	@printf "Detected CPU: %s\n" "$$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf unknown)"
	@printf "CPU flags: %s\n" "$(CPU_CFLAGS)"
	@printf "CFLAGS: %s\n" "$(CFLAGS)"
	@printf "LDFLAGS: %s\n" "$(LDFLAGS)"
	@printf "CHAT_CFLAGS: %s\n" "$(CHAT_CFLAGS)"

metal_infer: $(TARGET)

infer: $(INFER_TARGET)

chat: $(CHAT_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	@$(MAKE) --no-print-directory print-build-config
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC)
	@$(MAKE) --no-print-directory print-build-config
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) $(LINENOISE_SRC) $(LINENOISE_HDR)
	@$(MAKE) --no-print-directory print-build-config
	$(CC) $(CHAT_CFLAGS) -framework Foundation $(CHAT_SRC) $(LINENOISE_SRC) -o $(CHAT_TARGET)

clean: archive-debug
	rm -f $(TARGET) $(INFER_TARGET) $(CHAT_TARGET) $(SHADER_AIR) $(SHADER_LIB)

archive-debug:
	@if [ -d debug ]; then \
		entries="$$(find debug -mindepth 1 -maxdepth 1 ! -name .archived -print)"; \
		if [ -n "$$entries" ]; then \
			dest="debug/.archived/logs-$$(date +%Y%m%d-%H%M%S)"; \
			mkdir -p "$$dest"; \
			find debug -mindepth 1 -maxdepth 1 ! -name .archived -exec mv {} "$$dest"/ \;; \
			echo "Archived debug contents to $$dest"; \
		fi; \
	fi

clean-venv:
	rm -rf $(BUILD_DIR)/.venv

distclean: clean-venv
	rm -f $(TARGET) $(INFER_TARGET) $(CHAT_TARGET) $(SHADER_AIR) $(SHADER_LIB)
	rm -rf debug

# Run targets
run: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --expert 0)

verify: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --expert 0 --verify)

fast: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --expert 0 --fast --verify)

bench: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --expert 0 --fast --benchmark)

moe: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --fast --moe)

moebench: $(TARGET)
	$(call RUN_ENGINE_BENCH,--layer 0 --fast --moe --benchmark)

full: $(TARGET)
	$(call RUN_ENGINE_BENCH,--fast --full --k 4)

fullbench: $(TARGET)
	$(call RUN_ENGINE_BENCH,--fast --full --k 4 --benchmark)

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	cd $(BUILD_DIR) && ./infer --prompt "Hello, what is" --tokens 20 --k 4

# Chat TUI targets (use: make chat)

build-chat: $(CHAT_TARGET)

chat-run: $(CHAT_TARGET)
	cd $(BUILD_DIR) && ./chat --k 4

api-smoke: $(INFER_TARGET)
	bash tests/test_api_smoke.sh

# Performance regression benchmark — iterates the model registry (installed models not
# opted out via "benchmark": false), runs the uniform spec per model in its default config,
# appends prefill/decode metrics to assets/api_perf_log.tsv. Separate from `make test`
# because it starts a real server per model and is minutes-long.
bench-api: $(INFER_TARGET)
	bash tests/bench_api.sh

# Compare the latest benchmark rows against prior commits and flag regressions.
bench-report:
	python3 tests/bench_report.py

cli-smoke:
	bash tests/test_flashchat_cli.sh

manage-smoke:
	bash tests/test_flashchat_manage.sh

chat-render-smoke: $(CHAT_TARGET)
	bash tests/test_chat_tui_render.sh

tool-template-smoke: $(INFER_TARGET)
	bash tests/test_tool_template_render.sh

cache-roundtrip-smoke: $(INFER_TARGET)
	bash tests/test_disk_cache_roundtrip.sh

quant-helper-smoke:
	python3 tests/test_flashchat_quant.py

tokenizer-export-smoke:
	python3 tests/test_tokenizer_export.py

native-qwen-compile-smoke: $(INFER_TARGET)
	bash tests/test_native_qwen_compile.sh

mpp-tensorops-smoke:
	bash tests/test_mpp_tensorops_compile.sh

mpp-tensorops-runtime-smoke:
	bash tests/test_mpp_tensorops_runtime.sh

mpp-tensorops-bench:
	bash tests/bench_mpp_tensorops.sh

mtp-config-smoke:
	bash tests/test_mtp_config.sh

model-add-config-smoke:
	bash tests/test_model_add_config.sh

model-edit-config-smoke:
	bash tests/test_model_edit_config.sh

model-quant-config-smoke:
	bash tests/test_model_quant_config.sh

test: cli-smoke manage-smoke chat-render-smoke tool-template-smoke cache-roundtrip-smoke quant-helper-smoke tokenizer-export-smoke native-qwen-compile-smoke mtp-config-smoke model-add-config-smoke model-edit-config-smoke model-quant-config-smoke api-smoke
