#!/bin/bash
set -e
git config --global user.email "benpearce@example.com"
git config --global user.name "Ben Pearce"
git config --global credential.helper "store --file /workspace/.git-credentials"
pip install -r requirements-runpod.txt -q
export HF_HOME=/workspace/hf_cache
echo "Setup complete. HF_HOME=/workspace/hf_cache"
