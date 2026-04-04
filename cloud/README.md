# RunPod Cloud Workflow

This directory contains scripts for running experiments on [RunPod](https://runpod.io)
with a persistent **network volume**.  The network volume outlives individual pods, so
model weights and the repo clone are downloaded once and reused across all sessions.

---

## Architecture

```
RunPod pod (ephemeral)          Network volume /workspace (persistent)
────────────────────────        ──────────────────────────────────────
/root/.gitconfig          →     /workspace/.gitconfig
/root/.cache/huggingface/ →     /workspace/.cache/huggingface/
/workspace/newfile/             /workspace/newfile/          ← repo clone
                                /workspace/.git-credentials  ← stored token
```

**What resets each pod:** everything under `/root` (the container filesystem).  
**What persists:** everything under `/workspace` (the network volume).

The setup script recreates the two symlinks above on every boot, so you only
enter git credentials once and model weights are never re-downloaded.

---

## First-time setup (new network volume)

### 1. Create the pod

In the RunPod UI:
- Attach a **Network Volume** (recommend ≥ 100 GB for 72B weights)
- Mount path: `/workspace`
- Template: any CUDA image with Python ≥ 3.10
- GPU: pick based on model size (A100 80 GB for 72B, smaller for 7B)

### 2. Open a terminal and clone the repo

```bash
git clone https://github.com/sfcoder/newfile.git /workspace/newfile
```

You will be prompted for your GitHub username and a Personal Access Token
(PAT with `repo` scope).  The token is saved to `/workspace/.git-credentials`
and will not be asked for again on any future pod using this volume.

### 3. Edit the setup script (one-time)

Open `/workspace/newfile/cloud/runpod_setup.sh` and set your real name, email,
and repo URL at the top:

```bash
GIT_USER_NAME="Alice Smith"
GIT_USER_EMAIL="alice@example.com"
REPO_URL="https://github.com/yourname/newfile.git"
```

Commit and push this change from your local machine so it's in the repo.

### 4. Run setup

```bash
bash /workspace/newfile/cloud/runpod_setup.sh
```

Optionally, register and download models in the same step:

```bash
bash /workspace/newfile/cloud/runpod_setup.sh Qwen/Qwen2.5-7B Qwen/Qwen2.5-72B
```

This will:
- Symlink the git config and HF cache from `/workspace`
- Install Python dependencies from `requirements.txt`
- Download and register any requested model weights into the volume cache
- Print a summary of what's ready

---

## Returning to an existing pod session

If the pod is already running (you haven't stopped it), nothing is needed —
just open a terminal and continue.

---

## Starting a new pod (volume already set up)

Each new pod needs one command to restore the symlinks and pull the latest code:

```bash
bash /workspace/newfile/cloud/runpod_setup.sh
```

That's it.  Model weights are already on the volume; pip installs are fast
because the packages are cached.  No credentials to re-enter.

---

## Running experiments

All experiment scripts live in the repo root and `tools/`.

```bash
cd /workspace/newfile

# Threshold study (how sparse can masks be before verification fails?)
python3 threshold_study.py --model Qwen/Qwen2.5-7B

# Adversarial study (can an attacker substitute a cheaper model?)
python3 adversarial_study.py

# Max attacker savings (how many attention layers can be skipped?)
python3 tools/max_savings_test.py --model Qwen/Qwen2.5-7B

# Per-layer sparsity fingerprint
python3 tools/self_consistency.py
python3 tools/small_vs_large.py
```

Results are written to `analysis_results/<experiment>/` as JSON files.

---

## Pushing results back to GitHub

When an experiment finishes, push the results with a single command:

```bash
bash cloud/push_results.sh "72B threshold sweep — 5 thresholds, 10 prompts each"
```

This script:
1. Runs `migrate.py` to import any `standard_v1` JSON files into `results.db`
2. Checks that `results.db` is **not** staged (it's gitignored and stays local)
3. Stages everything under `analysis_results/`
4. Commits with your message and pushes to the current branch

---

## Updating the repo mid-session

If you push changes from your local machine and want to pull them on the pod:

```bash
git -C /workspace/newfile pull
```

Or just re-run the setup script — it always pulls the latest:

```bash
bash /workspace/newfile/cloud/runpod_setup.sh
```

---

## Cost tips

- **Stop (don't terminate) the pod** between sessions to keep the volume attached
  without paying for GPU time.
- The network volume itself has a small storage cost (~$0.07/GB/month on RunPod).
- 72B model in fp16 ≈ 144 GB.  In 4-bit quant ≈ 37 GB.

---

## Troubleshooting

**`git push` asks for credentials every time**  
The credential file may not have been created yet, or `/workspace` is a
different volume.  Run `bash cloud/runpod_setup.sh` and then do one manual
push — credentials will be saved.

**`ModuleNotFoundError` for a package**  
Re-run `pip install -r requirements.txt` from `/workspace/newfile`.  This
happens if the pod was updated or a new dependency was added.

**HuggingFace download fails / model not found**  
Some models require accepting a licence on huggingface.co.  Log in once:

```bash
huggingface-cli login
```

Then re-run `bash cloud/runpod_setup.sh Qwen/Qwen2.5-72B`.

**Out of disk space on the volume**  
Check what's using space:

```bash
du -sh /workspace/*  /workspace/.cache/huggingface/hub/*
```

Delete unused model snapshots with `huggingface-cli delete-cache`.
