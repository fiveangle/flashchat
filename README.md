# README — What is this, and how is this possible?

## History

Back when Apple began its quest to free itself from reliance on CPU makers Motorola (68k), IBM (PowerPC), and Intel (x86) for the beating heart of its microcomputer offerings, it bet hard on the fledgling ARM ecosystem and its flexible licensing model — and the bet paid off. Beyond the obvious gains in raw compute and efficiency, the move quietly opened the door for Apple to enter the then-anemic handset market. With that came the freedom to pull functions historically off-loaded to 3rd-party vendors back in-house, including the dedicated SSD processors it had been buying from the likes of Samsung and Toshiba — now folded onto the die alongside the rest of the "A" series, and later the "M" series. A huge undertaking, but Apple was no stranger to ripping out the heart of its computers and replacing it: it had already done so 3 times in the previous 2 decades.

A major benefit of this consolidation was bringing ancillary subsystems under a processor Apple now owned end-to-end. Driven primarily by cost reduction, pulling NAND die control directly onto the main CPU was one of those moves. It came at the price of easy storage upgrades, but offered something else besides not paying a 3rd party for their SSD management processor: S P E E D

What Apple has coined "Apple Fabric" — a fancy name for the CPU performing all the NAND management traditionally reserved for a specialized controller like those from SandForce (prior to its purchase by Seagate), Samsung, Marvell, and others — is essentially raw NAND wired as close as possible to the PCI bus so the CPU can drive it directly. The result is near-wire-speed storage.

## How It Changes Everything

Without an intermediary SSD controller in the path, the Apple M-series CPU in today's "Apple Silicon" systems pulls data from NAND faster than was ever possible with a separate processor in the way. It can never quite match RAM, but — much like Intel's all-but-defunct Optane and the more contemporary CXL "memory" — it gets close enough to be genuinely useful as a tier just below it.

## What Is This Project

This project, envisioned by the original author Dan Woods, is the answer to the question, "What if, instead of loading an entire Large Language Model into RAM the traditional way, we streamed the individual LLM layers directly to the GPU from storage?" Before Apple Fabric became ubiquitous across Apple's lineup — including the base-model MacBook Pro M5 I am typing this on — the idea was unheard of outside of esoteric non-volatile storage systems.

Using roughly 6GB of memory-resident space for the LLM metadata that needs low-latency access, the 60 "expert" layers are streamed on demand from Apple Fabric storage straight to the GPU as each layer is activated. At over 200GB on disk, the Qwen3.5-397B-A17B-4bit LLM has historically been reserved for big-iron systems. It works by holding 60 experts at the ready, each roughly 3.5GB, and for every token activates only the subset the prompt actually needs — streaming just that expert's data from the SSD to the GPU on the fly.

The result? Over 3.5 tok/sec on a base-model MacBook Pro M5 — a feat unheard of until now.

## Getting Started

Below is Dan Woods' original readme and paper, which I highly recommend you read. They offer a fascinating peek into the ingenuity of the human mind, as well as a wonderful demonstration of the capability multiplier machine learning brings to seeing that ingenuity through to fruition.

Dan's original project was polished just enough to publish his whitepaper, but it wasn't ready for the public to actually run, with fundamental rough edges around expert extraction and the tokenizer. This project closes that gap so the average Joe can set up and run flashchat on their own laptop without much fanfare. Using Opencode and the stealth coding-specific machine learning model Big Pickle, within a few hours I was able to turn this academic effort into a (hopefully) turn-key project you can run yourself.

You can find more details in RUN.md (written entirely by Big Pickle under my guidance), but the meat of it should be to simply:
```
git clone https://github.com/fiveangle/flashchat
cd flashchat
./flashchat
```

And follow the prompts to set up and run.

## Model Support

I've extended the original project to support a full configuration system with provisions to support multiple models.

Currently supported models:

* [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) - 37GB (default)
* [mlx-community/Qwen3.5-397B-A17B-4bit](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit) - 417GB

## Requirements
* Apple Silicon computer with at least 16GB of RAM (for 397B model) or 8GB of RAM (35B model)
* The listed amount of free space on the Apple Fabric SSD built into your Mac for the model you wish to use
* Xcode Command Line Tools

## Dan Woods' Original FLASH-MOE Project Info

[flash-moe README](./CLAUDE.md)

[flash-moe Whitepaper](./paper/flash_moe.pdf)
