# CS640 — Homework & Lab Notebooks (Fall 2025)

Hands-on notebooks for **Boston University CS 640** covering:
- **Search & planning** (Snake path planning)
- **Markov Reward Processes (MRP)** and **Markov Decision Processes (MDP)** (“Random Cats” / “Hungry Cats”)
- **MDP for Maze Solving**
- **GPU vs CPU** exercises and a simple **PyTorch** classifier sketch

> ⚠️ **Academic Integrity**  
> This repo is for learning and portfolio purposes. If you’re currently taking CS640, **do not copy** from here. Follow your course’s collaboration and citation rules.

---
## Contents

CS640_HomeworkAndLab/
├─ Homework/
│  ├─ CS_640_Homework_3_Planning_with_Search.ipynb   # Plan minimal moves in a Snake-like grid
│  ├─ CS_640_Homework_4_Random_Cats.ipynb            # Build an MRP over a 32×32 grid; cat finds fish
│  └─ CS_640_Homework_5_Hungry_Cats.ipynb            # Extend to MDP with actions (Up/Down/Left/Right)
└─ Lab/
├─ lab_4.ipynb                                     # CPU vs GPU timing; 1000-class classifier sketch
└─ lab_5.ipynb                                     # MDP for shortest-path maze solving

> Housekeeping files like `.DS_Store`, `__MACOSX/`, and IDE configs were removed/are ignored.

---

## Quick Start

### 1) Create an environment

Python **3.10+** recommended.

**Conda**
```bash
conda create -n cs640 python=3.10 -y
conda activate cs640

pip + venv

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

2) Install dependencies

These notebooks mainly use NumPy, Jupyter, and (for labs) PyTorch.

pip install jupyter numpy matplotlib
# Optional (GPU/CPU, per your machine):
# See https://pytorch.org/get-started/locally/ for the exact install line
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # Example for CUDA
# Or:
# pip install torch torchvision torchaudio  # CPU-only

3) Run Jupyter and open notebooks

jupyter notebook

Open the desired .ipynb under Homework/ or Lab/.

⸻
```
Notebook Summaries

Homework
	•	HW3 — Planning with Search
Implement a planner to guide a Snake-like agent to food in minimum steps.
Focus: state representation, legal moves, shortest path planning.
	•	HW4 — Random Cats (MRP)
Define a Markov reward process on a 32×32 grid. Model random-walk transitions and terminal behavior.
Focus: transition matrix P, reward specification, evaluating returns.
	•	HW5 — Hungry Cats (MDP)
Extend HW4 to a Markov decision process with 4 actions (Up/Down/Left/Right).
Focus: policy/value computation, optimal control on a grid.

Labs
	•	Lab 4 — CPU vs GPU; Simple Classifier
Use %%timeit to compare CPU/GPU ops and sketch a three-layer 1000-class classifier in PyTorch.
Focus: performance profiling, tensor shapes without bias, ReLU pipeline.
	•	Lab 5 — MDP for Maze Solving
Given a randomly generated maze n×n, use value iteration and policy extraction to find a shortest path from (0,0) to (n-1,n-1).
Focus: MDP setup, value iteration loop, policy improvement/extraction.

⸻

# GPU Notes
	##	Some cells are slow on CPU and fast on GPU. If you have CUDA:

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


	•	University clusters (e.g., SCC) may be required for larger runs; schedule access accordingly.

⸻

How to Work on the Notebooks
	1.	Read the instructions at the top of each notebook.
	2.	Complete TODOs where indicated (functions like value_iteration, plan, or transition matrix setup).
	3.	Run all cells to ensure check cells pass (some notebooks mention Gradescope).
	4.	Keep outputs (if the assignment requests it).

⸻

Testing & Reproducibility Tips
	•	Set seeds where possible:

import numpy as np, random, torch
np.random.seed(42); random.seed(42)
if 'torch' in globals(): torch.manual_seed(42)


	•	For notebook diffs, consider nbstripout (strip metadata) or export to .py via jupyter nbconvert --to script.

⸻

Project Hygiene

Add a .gitignore like:

# OS / IDE
.DS_Store
__MACOSX/
.idea/
*.iml

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python
__pycache__/
*.pyc
.venv/
.env/


⸻

License

No license granted for reuse. Educational viewing only.
If you want to adapt any portion, please ask for permission and credit the author.

⸻

Acknowledgments

Course scaffolding and problem statements inspired by BU CS 640 materials.

---

If you want, I can also generate the `.gitignore` and a minimal `environment.yml` for Conda so you can commit them alongside the README.