# README — What is this, and how is this possible?

## History

Back when Apple began its quest to free itself from reliance on CPU makers Motorola (68k), IBM (PowerPC), and Intel (x86) for the beating heart of its microcomputer offerings, it bet hard on the fledgling ARM ecosystem and its flexible licensing model — and the bet paid off. Beyond the obvious gains in raw compute and efficiency, the move quietly opened the door for Apple to enter the then-anemic handset market. With that came the freedom to pull functions historically off-loaded to 3rd-party vendors back in-house, including the dedicated SSD processors it had been buying from the likes of Samsung and Toshiba — now folded onto the die alongside the rest of the "A" series, and later the "M" series processors. A huge undertaking, but Apple was no stranger to ripping out the heart of its computers and replacing it: it had already done so 3 times in the previous 2 decades.

A major benefit of this consolidation was bringing ancillary subsystems under hardware control Apple now owned end-to-end. Driven primarily by cost reduction, pulling NAND die managment directly onto the main CPU was one of those moves. It came at the price of modular storage upgrades, but offered something else besides not paying a 3rd party for their SSD management processor: S P E E D

What Apple has coined "Apple Fabric" — a fancy name for the Apple Silicon CPU performing all the NAND management traditionally reserved for a specialized controller like those from SandForce (prior to its purchase by Seagate), Samsung, Marvell, and others — is essentially raw NAND wired as close as possible to the PCI bus so the CPU can drive it directly. The result is near-wire-speed storage.

## How It Changes Everything

Without an intermediary SSD controller in the path, the Apple M-series CPU in today's "Apple Silicon" systems pulls data from NAND faster than was ever possible with a separate processor in the way. It can never quite match RAM, but — much like Intel's all-but-defunct Optane and the more contemporary CXL "memory" — it gets close enough to be genuinely useful as a tier just below it.

## What Is This Project

This project, envisioned by the original author Dan Woods, is the answer to the question, "What if, instead of loading an entire Large Language Model into RAM the traditional way, we streamed the individual LLM layers directly to the GPU from storage?" Before Apple Fabric became ubiquitous across Apple's lineup — including the base-model MacBook Pro M5 I am typing this on — the idea was unheard of outside of esoteric non-volatile storage systems.

Using roughly 6GB of memory-resident space for the LLM metadata that needs low-latency access, the 4-10 of the 60 "expert" layers are streamed on-demand from Apple Fabric storage straight to the GPU as each layer is activated. At over 200GB on disk, the Qwen3.5-397B-A17B-4bit LLM has historically been reserved for big-iron systems. It works by holding 60 experts at the ready, each roughly 3.5GB, and for every token activates only the subset of them that the prompt actually needs — streaming just that expert's data from the SSD to the GPU on the fly.

The result? Over 3.5 tok/sec on a base-model MacBook Pro M5 — a feat unheard of until now.

## Getting Started

Below are links to Dan Woods' original readme and paper, which I highly recommend you read. They offer a fascinating peek into the ingenuity of the human mind, as well as a wonderful demonstration of the capability multiplier machine learning brings to seeing that ingenuity through to fruition.

Dan's original project was polished just enough to publish his whitepaper, but it wasn't ready for the public to actually run, with fundamental rough edges around expert extraction and real errors with the tokenizer. This project closes that gap so the average Joe can set up and run flashchat on their own laptop without much fanfare.

You can find more details in RUN.md, but the meat of it is to simply:
```
git clone https://github.com/fiveangle/flashchat
cd flashchat
./flashchat
```

…and follow the prompts to set up and run.

## Features
I've extended the original project from a reasearch project to usable system including:
* TUI driven setup wizard
* Menu-driven configuration system
* OpenAI-compatible v1 API endpoints with true tool calling support:

      POST /v1/chat/completions
      POST /v1/responses
      GET  /v1/models
      GET  /health
- Persistant prompt caching
- MTP support (tho MoE architecture does not lend itself well to MTP)
- Full server-side API debugging capability (speed metrics, full HTTP payload dumping, timing, etc)
- Model offload management (allows keeping runtime-only artifacts on Apple Fabric SSD while original weights and files can be stored on nearline USB/NAS)

## Model Support

* [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B) q4, q8 - 52GB (default)
* [Qwen/Qwen3.5-80B-A3B](https://huggingface.co/Qwen/Qwen3.5-80B-A3B) q4, q8 - 92GB
* [mlx-community/Qwen3.5-397B-A17B-4bit](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit) q4 - 417GB

## Requirements
* Apple Silicon computer with at least 24GB of RAM (for Qwen 397B MoE model with 4 active experts, 8=default) or 12GB of RAM (Qwen 35B MoE model, 8 active experts)
* Enough free space on the internal Apple Fabric SSD built into your Mac to hold the entire original model + experts extractions for the model you wish to use
* Xcode Command Line Tools

## Dan Woods' Original FLASH-MoE Project Info

[flash-moe README](https://github.com/danveloper/flash-moe/blob/main/CLAUDE.md)

[flash-moe Whitepaper](https://github.com/danveloper/flash-moe/blob/main/paper/flash_moe.pdf)
