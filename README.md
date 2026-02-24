# CIL Intrusion Detection (Adaptive IDS)

Cybersecurity project for year **2025/2026** at **University of Bologna (UniBo)**.

This repository implements an **adaptive Intrusion Detection System (IDS)** based on **Class-Incremental Learning (CIL)**, a Continual Learning setting where a model:
- is trained **sequentially** on a stream of tasks,
- **learns new attack classes over time** (including *0-day* / newly observed attacks),
- must mitigate **catastrophic forgetting** (performance degradation on previously learned classes after learning new ones).

## Project goals

- Build an IDS CIL scenario in **PyTorch**
- Integrate and compare strategies to mitigate catastrophic forgetting:
  - **iCaRL**
  - **ER (Experience Replay)**
  - **DER (Dark Experience Replay)**
- Evaluate how the **incremental task size** affects performance by testing different class-arrival scenarios (e.g. `1+1+1+...` vs `5+5` vs `2+3+5`), while **benign traffic is always present**.

---

## Repository structure (high level)

```txt
cil-intrusion-detection/
├─ main.py
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ utils.py
│  ├─ metrics.py
│  ├─ iCarl
│  ├─ er.py
│  └─ der.py
├─ data/
│  ├─ raw/            # original datasets (not tracked in git)
│  └─ processed/      # preprocessed datasets (optional)
├─ notebooks/         # preprocessing + experiments
├─ results/           # saved outputs (training logs, confusion matrices)
└─ artifacts/         # report/presentation material
```

---

## Datasets

This project uses network intrusion datasets such as:
- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html  
  (also available on Kaggle in various formats)
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

The repository expects datasets to be placed under `data/` (see next section).

---

## Setup & running

### Option A — Run on Google Colab (recommended)
You can run the project in Colab by uploading/cloning the repo, then executing `main.py` with your chosen configuration.

Example commands used in Colab:

```bash
!python main.py \
  --strategy der \
  --dataset 2015 \
  --scenarios 1+1+1+1+1+1+1+1 2+2+4 4+4 \
  --epochs 5 \
  --memory_size 5000 \
  --batch_size 256 \
  --lr 1e-3
```

```bash
!python main.py \
  --strategy der \
  --dataset 2017 \
  --scenarios 1+1+1+1+1+1+1 2+2+3 3+4 \
  --epochs 5 \
  --memory_size 5000 \
  --batch_size 256 \
  --lr 1e-3
```

### Option B — Run locally
#### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
# .venv\Scripts\activate    # (Windows PowerShell)
```

#### 2) Install dependencies
This repo currently **does not include a `requirements.txt`**.

Install at least:
- `torch` (choose the correct command from https://pytorch.org/)
- common scientific stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

Example (may need adjustments depending on your code and OS/CUDA):
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
# plus PyTorch from pytorch.org
```

#### 3) Prepare the data
Place datasets under `data/` according to the expected layout. From the existing repo notes:

- `data/raw/`: original datasets (not tracked in git)
- `data/processed/`: preprocessed datasets (optional)

The README notes that processed zips may unpack into a structure like:

```txt
data/processed/
└─ 2017/
   ├─ train/   # one file for each class
   └─ set/
```

You can also use the preprocessing notebooks in `notebooks/` (e.g., `cleaning_2015.ipynb`, `cleaning_2017.ipynb`) to generate the processed format.

#### 4) Run training / evaluation
From the repository root:

```bash
python main.py --strategy der --dataset 2017 --scenarios 1+1+1+1+1+1+1 2+2+3 3+4 --epochs 5 --memory_size 5000 --batch_size 256 --lr 1e-3
```

---

## Key CLI arguments (used in examples)

- `--strategy`: continual learning strategy (e.g. `icarl`, `er`, `der`)
- `--dataset`: dataset selection (e.g. `2015`, `2017`)
- `--scenarios`: one or more incremental task schedules (e.g. `1+1+1+1+...` or `2+2+4`)
- `--epochs`: epochs per incremental phase/task
- `--memory_size`: replay buffer size (for ER/DER-like methods)
- `--batch_size`: minibatch size
- `--lr`: learning rate
