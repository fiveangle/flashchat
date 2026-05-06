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
#   make test      — run all functional smoke tests
#   make help      — list available targets
#   make clean     — remove build artifacts
#   make archive-debug — archive repo-local debug contents under debug/.archived
#   make clean-venv — remove Python setup virtual environment
#   make distclean — remove build artifacts, repo-local debug, and setup venv
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

BUILD_DIR = metal_infer

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

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

.PHONY: all clean archive-debug clean-venv distclean help run verify bench moe moebench full fullbench fast metallib metal_infer infer chat build-infer infer-run chat-run build-chat api-smoke cli-smoke manage-smoke tool-template-smoke test

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
	@printf "  make tool-template-smoke  Run native tool template render/parser smoke test\n"
	@printf "  make api-smoke     Run HTTP API smoke test\n"
	@printf "  make test          Run all functional smoke tests\n"
	@printf "\n"
	@printf "Maintenance:\n"
	@printf "  make clean         Remove build artifacts and archive repo-local ./debug contents\n"
	@printf "  make archive-debug Archive repo-local ./debug contents under debug/.archived\n"
	@printf "  make clean-venv    Remove Python setup virtual environment\n"
	@printf "  make distclean     Remove build artifacts, repo-local ./debug, and setup venv\n"

metal_infer: $(TARGET)

infer: $(INFER_TARGET)

chat: $(CHAT_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) $(LINENOISE_SRC) $(LINENOISE_HDR)
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(CHAT_SRC) $(LINENOISE_SRC) -o $(CHAT_TARGET)

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

cli-smoke:
	bash tests/test_flashchat_cli.sh

manage-smoke:
	bash tests/test_flashchat_manage.sh

tool-template-smoke: $(INFER_TARGET)
	bash tests/test_tool_template_render.sh

test: cli-smoke manage-smoke tool-template-smoke api-smoke
