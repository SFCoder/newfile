# SwiGLU Proof-of-Inference Blockchain — Complete Project Handoff

## How to Use This Document
Upload this file at the start of any new Claude conversation to get the AI fully up to speed on the project. It contains everything needed: what was built, what was proven, what the theory is, and what needs to happen next.

The project's code, experiments, and results all live in a private GitHub repo: https://github.com/SFCoder/newfile.git (branch: claude/max-savings-experiment-Gktym). The results database can be queried with `python3 analyze.py --summary` for a quick snapshot of all experimental data.

---

## 1. CORE HYPOTHESIS & MECHANISM

### The Discovery
SwiGLU-based LLMs (like the Qwen 2.5 family) produce deterministic neuron activation patterns at the FFN (Feed-Forward Network) layer. At each layer, a "gate" determines which neurons fire — producing a binary mask of active vs inactive neurons. This mask is:
- **Deterministic**: Same model + same input = same mask, every time
- **Input-dependent**: Different prompts produce different masks
- **Model-specific**: Different models produce statistically distinguishable masks
- **Sparse at scale**: The 72B model has 20-29% of neurons inactive (vs 3% for the 7B)

### The Insight
These neuron masks can serve as **proof artifacts** for AI inference. A provider who claims to have run a specific model on a specific prompt must produce a mask that's consistent with that model's behavior. The mask can't be forged without actually running the model, because:
- Random masks produce rank ~10,558 predictions (vs rank ~1 for real masks)
- Wrong-model masks are statistically distinguishable (Cohen's d = -2.0)
- The mask encodes genuine information about the computation

### The Verification Mechanism
Given a provider's claimed output and neuron mask:
1. The verifier loads the same model weights (verified by SHA-256 hash)
2. Replays only the active neurons (sparse replay) — much cheaper than full inference
3. Checks that the sparse replay produces the same output
4. If it matches, the computation was honest. If not, the provider cheated.

### The Split Verification Breakthrough
The FFN computation at each layer decomposes by neuron. Each neuron's contribution to the layer output is independent — they combine through addition. This means:
- You can split the 29,568 neurons across 1,000+ validators
- Each validator computes their assigned ~30 neurons
- The partial results sum to the exact same answer as computing all neurons on one machine
- **This is mathematically exact, not approximate** (proven experimentally)

The same property applies to attention heads. Each of the 64 attention heads computes independently, and their contributions through the output projection matrix combine through addition. Attention heads can be split across validators the same way neurons can.

---

## 2. WHAT WAS BUILT

### Verification System (Python)
All code is in the GitHub repo root:
- `model_registry.py` — Weight-verified model loading with SHA-256 hash checking, 4-bit quantization support
- `verifier.py` — Pure verification function, no server dependencies
- `provider.py` — Inference with automatic neuron mask capture
- `verification_api.py` — FastAPI wrapper for the verification engine
- `demo.py` — 6 demo scenarios (honest, tampered-output, wrong-model, fake-masks, tampered-weights, different-model)
- `registry.json` — Tracks Qwen2.5 0.5B, 3B, 7B, 72B with weight hashes
- Tests: 33 verification + 36 registry tests, all passing

### Adversarial Testing Suite
Structured for future reuse against a live Cosmos blockchain:

```
adversarial_suite/
  attacks/
    base.py          — AttackResult dataclass
    attention_skip.py — AttentionSkipAttack (implemented)
    model_substitution.py — stub
    random_mask.py    — stub
    token_tamper.py   — stub
  verification/
    base.py           — VerificationTarget ABC
    local.py          — LocalVerification (wraps verifier.py)
    cosmos.py         — CosmosVerification stub (for future on-chain testing)
  metrics/
    compute.py        — Pure metric functions (token_match_rate, cosine_similarity, etc.)
    reporting.py      — Tables, CSVs, plots
  db/
    schema.py         — SQLite schema (6 tables: experiments, hardware, models, prompts, configurations, results)
    writer.py         — ResultsWriter context manager (auto-detects hardware, captures git info)
    standard_format.py — Universal JSON format for experiment results
```

### Experiment Scripts
In `tools/`:
- `max_savings_test.py` — Quantifies maximum attacker advantage from attention skipping
- `attention_split_test.py` — Tests attention head splitting across hardware
- `export_dashboard.py` — Generates interactive HTML dashboard from results.db

In repo root:
- `split_verification_test.py` — Tests FFN neuron splitting correctness
- `cross_hardware_test.py` — Compares MPS vs CUDA computation
- `attention_trust_test.py` — Tests whether fake attention produces coherent output
- `threshold_study.py` — Sweeps neuron thresholds across models
- `adversarial_study.py` — Tests model substitution, random masks, token tampering
- `sparsity_layers.py` — Per-layer sparsity analysis

### Data Analysis Tools
- `analyze.py` — CLI query tool for the results database
  - `python3 analyze.py --summary` — Dashboard of all experiments and key findings
  - `python3 analyze.py --metric token_match_rate --group-by model_name,attack_type`
  - `python3 analyze.py --compare model_name --metric savings_pct --plot`
  - `python3 analyze.py --list-experiments`
  - `python3 analyze.py --export results.csv --where "attack_type='attention_skip'"`
- `migrate.py` — Imports all JSON results into the database (idempotent)
- Interactive HTML dashboard generated by `python3 tools/export_dashboard.py`

### Cloud Infrastructure
In `cloud/`:
- `runpod_setup.sh` — Idempotent setup for RunPod: git config, credential store, repo clone/pull, HF cache symlink, pip install, model registration
- `push_results.sh` — Runs migrate.py, stages analysis_results, commits and pushes
- `README.md` — Full workflow documentation

### GitHub Repo
- Private repo: https://github.com/SFCoder/newfile.git
- Branch: claude/max-savings-experiment-Gktym
- RunPod network volume (100GB) for persistent model storage
- Git credential store on network volume for passwordless push

---

## 3. EXPERIMENTAL RESULTS

### Database Summary (as of last session)
- 11 experiments, 4 models (0.5B, 3B, 7B, 72B), 26 prompts, 8,465 total results
- Run `python3 analyze.py --summary` for current state

### Key Findings

**FFN Split Verification (Experiment 1) — PROVEN**
- Neuron computation splits across any number of machines and sums to exact same result
- Column slicing and zero-and-multiply produce identical results (zm_vs_cs = 0.000000)
- Float32 accumulation reduces worst-case error to 0.014
- Works at any granularity: 2 splits to 1,000 splits
- Token match 80% (the 20% failure is float16-vs-float32 precision, not the split)

**Cross-Hardware Consistency (Experiment 2) — UNDERSTOOD**
- Per-layer differences between Apple Silicon MPS and NVIDIA CUDA are tiny (0.001-0.008)
- Differences accumulate to large values (up to 18) by layer 27
- Token match 80% across hardware
- KEY INSIGHT: Layer-by-layer verification prevents accumulation because each layer's input is committed rather than independently computed

**Attention Trust Hypothesis (Experiment 4) — FAILED (important finding)**
- Fake attention produced coherent output 47% of tests
- Wrong-prompt attention: 7/15 coherent. Zero attention: 7/15 coherent
- Fake FFN much worse: only 3/15 coherent (17% match)
- CONCLUSION: Cannot trust attention without verification. But fake FFN is more damaging, confirming FFN verification is core.
- HOWEVER: Single-layer tolerance doesn't translate to multi-layer exploitation (see max savings)

**Max Savings / Attacker Advantage — BOUNDED AT 3.54%**
- Maximum attacker advantage from skipping attention at multiple layers: 3.54%
- Complex prompts: 2.4% (2 layers skippable). Moderate: 3.5% (3 layers). Simple: 0%
- Reconciliation with Experiment 4: individual layer tolerance doesn't translate to multi-layer skipping because damage compounds
- This means attention skipping is NOT a consensus-threatening vulnerability

**Attention Split Verification (Experiment 7) — PROVEN**
- Attention heads split correctly with 0.99999+ cosine similarity on both MPS and CUDA
- Cross-hardware: 100% token match, mean cosine 0.99999890
- Per-head worst case: 0.99996560
- BETTER cross-hardware consistency than FFN
- Validator cost: 9.73 MB per layer (1 attention head + 100 FFN neurons)

**Earlier Experiments**
- Fingerprint Classification (0.5B vs 3B): 100% accuracy, 0% self-confusion, Cohen's d = -2.0
- Sparse Replay (7B): 5/5 exact text matches, cosine 0.9999+
- 72B Threshold Study: 29% compression at threshold 0.1, 20.4% at 0.05. Pass rate 84.8%-100%
- 72B Random Mask Attack: Real masks rank 1.02, random masks rank 10,558
- 72B Per-Layer Sparsity: Layers 0-6: 99.4-99.9% inactive. Deep layers show real sparsity unlike 7B
- 72B Adversarial Study: Model substitution detectable (honest rank 1.00, fraud rank 1.38-2.41). Token swap: even 1 token produces rank 5,215

---

## 4. CONSENSUS MECHANISM THEORY

### Architecture: Layer-as-Block with Self-Assembling Sub-Blocks

Each inference is verified layer-by-layer. At each layer:
- FFN neurons are split across validators (29,568 pieces possible)
- Attention heads are split across validators (28-64 pieces)
- Both are proven mathematically exact and cross-hardware consistent
- No single block producer — blocks self-assemble from validator partial results

Each validator is permanently assigned specific weight columns (~10MB download), then performs milliseconds of computation per verification. The assignment is permanent for a given model — load once, verify indefinitely.

### Two-Sided Mining Model
- **Providers** earn coins for inference work (expensive computation)
- **Verifiers** earn coins for verification work (cheap computation)
- Both contribute to chain security
- An attacker needs to control 51% of BOTH inference AND verification throughput
- At scale, this is enormously expensive because inference dominates the security weight
- The coin split ratio between providers and verifiers is a tunable parameter that affects security properties

### Sybil Resistance Argument (Theoretical — Not Fully Proven)
At high throughput, the total verification work is enormous. Controlling 51% requires matching half the network's continuous computation. Security scales with demand — more inferences = more verification work = higher attack cost.

Key insight: if providers earn mining rewards for inference work, the attacker needs 51% of total computational contribution (inference + verification), not just 51% of the cheap verification layer. Since inference is expensive, this dramatically raises the attack cost.

Additional defense: randomly saving verified inferences from past blocks to re-verify in future blocks creates compounding security — each new block implicitly confirms past blocks.

IMPORTANT CAVEAT: This argument is strongest at high volume. During low-volume bootstrapping, the security is weak (same problem Bitcoin had in 2009).

### Ledger Binding Problem — IDENTIFIED, NOT SOLVED
Critical gap: inference proofs aren't derived from transaction data (unlike Bitcoin where the hash is derived from block contents). An attacker could reuse a valid inference proof with different transactions.

Options identified:
- (A) Include transaction hash in the prompt
- (B) Block producer signs binding
- (C) Validators attest to transaction state alongside their partial results
- (D) Inference output determines transaction selection (most Bitcoin-like)

This is the most important unsolved problem. Without ledger binding, the computation proves work happened but doesn't secure the transaction history.

### Trust-Attention Architecture
Originally proposed to simplify verification by only verifying FFN (not attention). The idea: the provider submits attention outputs at each layer, validators take them as given and only verify FFN.

STATUS: The approach works as a practical verification system (max attacker advantage bounded at 3.54%). It was shown that:
- Single layers can tolerate fake attention (Experiment 4)
- But exploiting this across multiple layers simultaneously yields only 2.4-3.5% savings (max savings test)
- This is negligible for consensus security

However, attention CAN be verified through head splitting (Experiment 7), so the protocol can optionally verify both FFN and attention for maximum security.

---

## 5. EIGHT CONSENSUS PROBLEMS

These must be solved to go from "verification mechanism" to "consensus mechanism":

### Problem 1: Sybil Resistance — PARTIALLY ADDRESSED
The volume-based security argument (attacker needs 51% of total computational throughput) is promising. The two-sided mining model (providers + verifiers) raises the bar because inference is expensive. But formal proof is needed, and bootstrapping (low-volume) period remains vulnerable.

### Problem 2: Block Producer Selection — POTENTIALLY ELIMINATED
The self-assembling sub-block architecture may eliminate the need for block producer selection entirely. Validators write their partial results into sub-blocks; blocks emerge from aggregate activity. No single entity controls block assembly.

Remaining concern: without a block producer, who determines the time boundary for each block? And the sub-block corruption vulnerability — a minority attacker controlling all validators for a specific neuron group could corrupt that sub-block. Defense: overlapping random assignments (each neuron verified by 2-3 independent validators).

### Problem 3: Finality — UNSOLVED
Depends on solving Problems 1 and 2. If validators prove identity through computation, voting-based finality becomes possible. If cumulative computation serves as security (like PoW), finality is probabilistic.

### Problem 4: Fork Choice Rule — UNSOLVED
Depends on finality model. If absolute finality (like PoS), forks don't persist. If probabilistic (like PoW), need "cumulative verified computation" metric.

### Problem 5: Liveness — PARTIALLY ADDRESSED
Validator reassignment handles offline validators. Low-demand periods (no inference requests) remain a concern — padding inferences are wasteful but functional.

### Problem 6: Partition Tolerance — UNSOLVED
Requires understanding the finality and fork choice models first.

### Problem 7: Economic Security Quantification — PARTIALLY SOLVED
Experimental data provides the inputs: max attacker savings (3.54%), detection rates for various attacks, cross-hardware tolerances. Missing: formal cost-of-attack formula.

### Problem 8: Incentive Compatibility — PARTIALLY SOLVED
Major cheating strategies are bounded or caught. Missing: formal game-theoretic proof covering all participant roles and deviations.

---

## 6. REMAINING EXPERIMENTS

### Priority Research (in order)

**Experiment 6: Cross-Hardware Tolerance Exploitation** — Can an attacker hide fraud within the 0.001-0.008 per-layer hardware tolerance? Introduce deliberate errors within tolerance at each layer, check if they compound into meaningful output changes. Testable on MacBook with 7B.

**Experiment 8: Long Prompt Behavior** — Do results hold on 200-500 token inputs and complex reasoning prompts? Tests sparsity patterns, cross-hardware differences, split precision. Testable locally then cloud.

**72B Max Savings** — Confirm the 3.54% attacker bound on the larger model. One cloud session.

**72B Attention Split** — Confirm attention splitting at 72B scale. One cloud session.

**Per-Validator Minimal Cost (72B)** — Measure actual computation time and memory for a validator checking a few neurons + heads on the 72B model. Confirms cheap hardware participation.

### Theoretical Work Needed

**Sybil Resistance Formalization** — Formal model of the two-sided mining security. Calculate: at X inferences per minute on a Y-parameter model, what does 51% of computational throughput cost? Compare to value flowing through the chain.

**Ledger Binding Design** — Choose and formalize the mechanism for binding inference proofs to transaction state. Option D (inference output determines transaction selection) is most promising.

**Security Model** — Formal cost-of-attack formula incorporating all experimental data.

---

## 7. KEY STRATEGIC DECISIONS

- **Layer 1 blockchain** (not Layer 2). Cosmos SDK chosen for sovereignty and IBC interoperability.
- **Stealth mode** until patent filed. No public disclosure of mechanism.
- **Patent filing** before any public activity. Provisional patent ~$200-3,000.
- **Open model support** (HuggingFace Transformers) with curated initial set.
- **Directory-style marketplace** for MVP (not automated matching) — providers listed, users click to select. Works at any scale including 1 provider.
- **Proof of stake NOT required** if volume-based Sybil resistance holds at scale — this is the key theoretical question being explored.

---

## 8. WORKFLOW

### Development Flow
1. Design experiments and write Claude Code prompts in Claude chat
2. Claude Code builds the code and pushes to GitHub
3. For local experiments: `git pull` on MacBook, run, results auto-save to JSON + database
4. For cloud experiments: spin up RunPod pod, attach network volume, `bash cloud/runpod_setup.sh [model]`, run experiment, `bash cloud/push_results.sh "description"`
5. Pull results to MacBook, `python3 migrate.py`, `python3 analyze.py --summary`
6. Regenerate dashboard: `python3 tools/export_dashboard.py`

### RunPod Setup
- Network volume: 100GB, persists across pods
- Any GPU with 16GB+ VRAM for 7B experiments
- A100 80GB or similar for 72B experiments
- Git credentials stored on network volume (`/workspace/.git-credentials`)
- Models cached on network volume (`/workspace/.cache/huggingface/`)

### User's Hardware
- MacBook Pro M-series, 24GB RAM, Apple Silicon (MPS)
- Python 3.11.9 via pyenv, venv at `/Users/benpearce/Documents/Experiment/newfile/.venv`
- RunPod account with network volume
- GitHub: SFCoder/newfile (private repo)

---

## 9. MARKETPLACE ANALYSIS

### Provider Tiers
- **Gamers** ($500-1,500 hardware): 7-14B models, 15-25 t/s. Viable as verifiers, not competitive as inference providers.
- **Enthusiasts** ($2,500-5,000): 70B models, 7-15 t/s. Core market for decentralized inference.
- **Labs/SMBs** ($10,000-50,000): 200-405B models, 20-60 t/s. Idle compute with near-zero marginal cost.
- **Professional** ($50,000+): Everything at high throughput. Competes with centralized providers.

### Centralized Competition
- Groq: $0.64/M tokens, 278 t/s (custom LPU chips)
- Cerebras: $0.60/M tokens, 450 t/s (wafer-scale engine)
- DeepInfra: $0.36/M tokens, ~80 t/s
- Together AI: $0.88/M tokens, ~100 t/s

### Key Insight
Individual gamers/enthusiasts can't compete on speed or price with centralized providers. Per-query savings are fractions of a cent. The value proposition is: verification (no centralized provider offers it), privacy, and censorship resistance. Token mining rewards (not inference fees) are the primary provider incentive.

---

## 10. NEXT SESSION QUICK START

To get productive immediately in a new conversation:

1. Upload this document
2. State what you want to work on (e.g., "Let's formalize the Sybil resistance model" or "Let's run Experiment 6")
3. For experiment results context, run: `python3 analyze.py --summary` and paste the output
4. For code context, the repo is at https://github.com/SFCoder/newfile.git on branch claude/max-savings-experiment-Gktym
