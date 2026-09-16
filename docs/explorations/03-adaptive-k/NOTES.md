# 03 — Adaptive expert count (routing weight mass)

Branch: `exp/03-adaptive-k` (from main@d507059)

## Hypothesis

Select the smallest top-K prefix (floor 5-6) covering FLASHCHAT_ADAPTIVE_K_MASS
of routed mass, renormalize; cuts expert bytes (16GB) + expert GPU compute.

## Method

Interleaved A/B/A/B quick-bench (100 tok x 1 run x 2 rounds — ambient machine
drift was +-6% this session, so interleaving is mandatory; see ../00-baseline/NOTES.md
protocol note). Temp-0 output comparison for quality.

## Observations

- Qwen3.6-35B routing is FLAT: at mass 0.95 avg K = 6.984; at mass 0.90/min5
  avg K = 6.921 (13.5% fewer expert reads). The 8th expert typically carries
  ~3% mass — cannot drop 2+ experts without mass <= 0.85.
- Interleaved decode (100 tok): off 16.55/16.39 vs m90 17.24/17.21
  => **+4.6%**.
- mass 0.95: +0.4% (K 6.98 — too little cut). mass 0.98: never triggers.
- Temp-0 quality: outputs near-identical for ~7 lines, then natural coherent
  divergence (expected — computation genuinely differs).

## Verdict

KEEP as opt-in (FLASHCHAT_ADAPTIVE_K_MASS, default off). +4.6% decode here;
worth ~13.5% of expert SSD bytes on 16GB. Not default-on: the project treats
exact computation as a feature (lossless MTP ethos); needs real-workload
quality validation (bench-api tool-call scenarios, acceptance sampling)
before considering default. If merged, wire the knob through the config
chain per AGENTS.md (menu + config.sh 8 sites + schema bump).
