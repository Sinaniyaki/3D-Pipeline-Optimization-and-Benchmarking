# Deep Comparison Between DVR and DeepSDF
*A Comparative Study on 3D Object Reconstruction (Final Project — CMPT 469)*

**Authors:** Sina MohammadiNiyaki & Ali Nikan  
**Course:** CMPT 469 — Computer Vision, Simon Fraser University (Spring 2025)  
**Instructor:** Prof. Manolis Savva

> **Note on scope:** Earlier plans (proposal/milestone) considered multiple datasets. The **final implementation and results** used **ShapeNet (chairs, `03001627`) only**. DTU remained out-of-scope for the final submission due to environment/data constraints.

---

## 🧠 Overview
We compare two implicit 3D reconstruction methods:

- **DVR — Differentiable Volumetric Rendering** (image-supervised via differentiable rendering)  
- **DeepSDF — Deep Signed Distance Functions** (3D SDF-supervised, mesh extracted via Marching Cubes)

We analyze trade-offs in **reconstruction quality** (Chamfer Distance, IoU), **runtime**, and **GPU memory** on a shared experimental setup using **ShapeNet (chairs)**.

---

## 🏁 Final Project Objectives
1. Implement reliable, reproducible environments for **DVR** and **DeepSDF**.  
2. Train/evaluate both on **ShapeNet chairs** with consistent metrics.  
3. Compare qualitative meshes and quantitative metrics.  
4. Document challenges (env/deps), fixes, and practical guidance for reproducing results.

*(Optimizations like sparse grids / hash encodings were explored during development but not fully productized in the final repo.)*

---

## 🏗️ Repository Structure

```
Deep-Comparison-Between-DVR-and-DeepSDF/
│
├── dvr/                     # Differentiable Volumetric Rendering (cleaned code/configs)
│   ├── configs/             # e.g., ours_depth.yaml (tuned for chairs & 8GB GPUs)
│   ├── src/                 # training/eval/generate scripts (see dvr README if present)
│   └── (no large outputs committed)
│
├── deepsdf/                 # DeepSDF (cleaned code + build configs)
│   ├── CMakeLists.txt
│   ├── model/               # training scripts (e.g., train_chair.py)
│   ├── preprocessing/       # preprocessing (e.g., preprocess_chairs.py)
│   ├── reconstruct/         # mesh extraction (e.g., reconstruct.py)
│   └── (no large outputs committed)
│
├── docs/
│   ├── Proposal.pdf
│   ├── Milestone Report.pdf
│   └── Final Report.pdf
│
└── README.md                # this file
```

> Large datasets, checkpoints, renders, and logs were **intentionally excluded** to keep the repo lean. See **Assets & Data** for external hosting notes.

---

## ⚙️ Environment (Final Setup Used)
| Component | DVR | DeepSDF |
|---|---|---|
| **Python** | 3.8 (venv, pip) | 3.8 (venv, pip) + C++17 |
| **Core libs** | PyTorch 1.7.1, torchvision 0.8.2, CUDA 11.0 | PyTorch + Pangolin, Eigen3, nanoflann |
| **GPU** | NVIDIA RTX 3070 | NVIDIA RTX 3070 |
| **OS** | Ubuntu (WSL2) | Ubuntu (WSL2) |

**Why venv (pip) over conda?** Faster activation and fewer CUDA/version conflicts in our tests.

---

## 📦 Setup & Installation

### 1) Clone
```bash
git clone https://github.com/Sinaniyaki/Deep-Comparison-Between-DVR-and-DeepSDF.git
cd Deep-Comparison-Between-DVR-and-DeepSDF
```

### 2) Python env
```bash
python3 -m venv env
source env/bin/activate   # Linux/Mac
# env\Scripts\activate  # Windows (if building DeepSDF, WSL is recommended)
```

### 3) Install dependencies

**DVR (Python):**
```bash
pip install torch==1.7.1 torchvision==0.8.2  # match CUDA 11.0 build
pip install numpy scipy matplotlib imageio trimesh pyyaml scikit-image tqdm
```

**DeepSDF (C++/Python mix):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev
# Pangolin & nanoflann installed per system; ensure C++17 and correct include paths
pip install numpy torch torchvision matplotlib scipy tqdm
```

> If Pangolin causes OpenGL GUI issues under WSL, use **CLI-only** workflows and ensure you compile with **C++17**. We patched API changes (e.g., `std::get`, `std::variant`) during our build.

---

## 📂 Assets & Data (ShapeNet Only)
Due to size limits, **datasets and generated assets are not in this repo**.

- **Dataset:** [ShapeNetCore.v2](https://shapenet.org/) — class `03001627` (*chairs*).  
- **DeepSDF preprocessing:** we sampled **~15,000 SDF points/mesh** and limited to a subset for runtime.  
- **DVR supervision:** depth-only configuration for chairs; RGB/color losses disabled in final runs to fit 8GB GPUs.

You can add a tiny sample (a few meshes) in an `assets/` folder for quick sanity tests.

---

## 🧪 Training & Evaluation (Final Runs)

### DVR (ShapeNet Chairs)
```bash
# from dvr root (after build if needed)
python train.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml
python generate.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml
# optional:
python eval_meshes.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml
```
- Edits to `ours_depth.yaml`:
  - `classes: ['03001627']`
  - lower batch size / points per iteration for 8GB GPU
  - disable normal/RGB losses for faster convergence

### DeepSDF (ShapeNet Chairs)
```bash
# build (first time)
mkdir -p build && cd build
cmake .. && make -j8
cd ..

# preprocess subset of chairs to SDF samples
python preprocessing/preprocess_chairs.py

# train (customized script with logging/profiling)
python model/train_chair.py

# reconstruct meshes via marching cubes
python reconstruct/reconstruct.py  # or reconstruct.py depending on layout
```

**DeepSDF example metrics (final submission run):**
- Average Chamfer Distance: **239.33**  
- Average IoU: **0.005**  
- Inference time: **~22.17 s** per shape  
- Peak GPU memory: **~684 MB**  
- Model size: **~23 MB**, params: **~1.8 M**, epochs: **~822**

> DVR metrics were computed with the project tools; exact numbers vary by config/hardware, so we report method and settings rather than fixed values.

---

## 📊 Final Comparison (ShapeNet Chairs)

| Model | Supervision | Pros | Cons |
|---|---|---|---|
| **DVR** | Multi-view images (depth-only in final) | Watertight meshes; strong geometric fidelity | Higher memory/compute; slower to train |
| **DeepSDF** | SDF samples (3D supervision) | Compact model; smooth surfaces; moderate memory | Fine detail weaker; slower inference; needs Marching Cubes |

*Hybrid/optimization ideas discussed in earlier reports were not shipped in the final repo.*

---

## 🧰 Troubleshooting Notes
- **CUDA mismatch:** pin `torch==1.7.1` + `CUDA 11.0` builds.  
- **DTU configs:** were prototyped but **not used** in final experiments.  
- **WSL OpenGL:** avoid GUI rendering; compile with C++17 and correct include/link flags.  
- **Dataset structure:** ensure ShapeNet paths match your preprocessing scripts.

---

## 📚 Documentation
All formal write-ups are included in `/docs`:

- `Proposal.pdf` — original plan and background  
- `Milestone Report.pdf` — mid-project progress and adjustments  
- `Final Report.pdf` — final scope, experiments (ShapeNet-only), and conclusions

---

## 🙏 Acknowledgements
Based on and inspired by:  
- **DVR:** Niemeyer et al., *Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision* (CVPR 2020)  
- **DeepSDF:** Park et al., *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation* (CVPR 2019)

Project completed for **CMPT 469 (Computer Vision), Simon Fraser University**.

---

## 👥 Authors
**Sina MohammadiNiyaki** — North Vancouver, BC — [LinkedIn](https://www.linkedin.com/in/sinamniyaki) · [GitHub](https://github.com/Sinaniyaki)  
**Ali Nikan** — Coquitlam, BC — [LinkedIn](https://www.linkedin.com/in/alinikan79) · [GitHub](https://github.com/alinikan)

---

## 📎 Citation
```bibtex
@misc{mohammadinikayi2024dvrdeepsdf,
  title={Deep Comparison Between DVR and DeepSDF},
  author={MohammadiNiyaki, Sina and Nikan, Ali},
  year={2024},
  institution={Simon Fraser University, CMPT 469: Computer Vision}
}
```
