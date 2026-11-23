# Pixel diffusion with self-flow and self-teaching, up to a JiT-style 256 px run

A pixel-space diffusion transformer (x-prediction, pyramid readouts). Two "the model teaches itself" ideas were found through automated FID loops: self-flow (an averaged copy of the model teaches early layers) and self-teaching (a frozen copy judges the output). The recipe was then scaled to a class-conditional 109M model at 256 px.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `unify3090/job/`, `job_selfteach/`, `job_stscale/`: the training code
- `unify3090/*/LOG.md`: every FID-loop test with time, change and score
- `unify3090/config_*.json`: run configs
- `colab-jit/job/`, `configs/`, `make_stages.py`: the 128 px → 256 px JiT-style run
- `RESEARCH_PRINCIPLES.md`: my instructions to the agent, distilled into 22 research principles

## Running it

The training code expects a CUDA GPU and ImageNet at 128 or 256 px. FID uses 10k samples and 16 Euler steps (not comparable to 50k-sample paper numbers).

## Results

20M model, 100k steps: FID 134.7 → 114.2 with self-flow + self-teaching. 100M model: 98.4 at 400k steps. Class-conditional 109M model at 256 px, guidance 3: FID 42.3 after 820k steps.

**Code:** written and run by Claude (Anthropic) as a coding agent, to my experiment specs. I chose the questions, the controls and the pass/fail rules, and read the results.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
