# ForestGuard — Exhaustive Technical Project Documentation

> **Audit basis:** Every claim in this document is derived from direct inspection of files under `D:\CDAC_PROJECT\DNN_Project\forestguard` as of the audit date. No features, metrics, or dependencies are invented. Where README prose diverges from executable code, both are cited and the discrepancy is flagged.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Inventory](#2-repository-inventory)
3. [System Architecture](#3-system-architecture)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Module Reference — Inputs, Outputs, and Mechanisms](#5-module-reference--inputs-outputs-and-mechanisms)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Technology Stack](#7-technology-stack)
8. [Configuration and Infrastructure](#8-configuration-and-infrastructure)
9. [Runtime Artifacts and On-Disk State](#9-runtime-artifacts-and-on-disk-state)
10. [Known Gaps, Defects, and Documentation Drift](#10-known-gaps-defects-and-documentation-drift)
11. [Technical Interview Questions](#11-technical-interview-questions)

---

## 1. Executive Summary

**ForestGuard** is a binary audio classification project that distinguishes **Wildlife** (label `0`) from **Threat** (label `1`) using mel-spectrogram features and a custom PyTorch CNN (`ForestGuardCNN`).

| Property | Verified value |
|---|---|
| **Primary entry point** | `forest.ipynb` (37 cells) |
| **Source dataset** | ESC-50 (`dataset/esc50.csv`, 2,000 rows; 50 categories) |
| **Curated subset** | 360 samples across 9 ESC-50 categories |
| **Feature tensor shape** | `(N, 128, 216)` float32 mel-spectrogram (dB scale) |
| **Train / validation split** | 288 / 72 (80/20, stratified, `random_state=42`) |
| **Model parameters** | 226,442 trainable (per notebook output); 226,927 total tensor elements in saved checkpoint |
| **Best validation accuracy** | 97.22% (`0.9722`, early-stopped at epoch 23) |
| **Deployment artifacts** | `models/forestguard_cnn.pt`, `models/label_map.json`, `models/norm_stats.npz` |
| **Live demo (Hugging Face)** | [forestguard_CDAC-DNN-Project Space](https://huggingface.co/spaces/priyansupattanaik/forestguard_CDAC-DNN-Project) — Gradio app at `https://priyansupattanaik-forestguard-cdac-dnn-project.hf.space` |
| **Inference app** | `app.py` — Gradio UI synced from Hugging Face Space (also loads artifacts from `models/`) |
| **Dependencies** | `requirements.txt` |
| **Auxiliary Python modules** | `dataset/bc_utils.py`, `dataset/utils.py`, `dataset/utils2.py` — present but **not imported** by `forest.ipynb` |

There is **no** `pyproject.toml`, `Dockerfile`, CI configuration, or test suite in the repository.

---

## 2. Repository Inventory

### 2.1 Top-level files

| Path | Type | Role | Referenced by |
|---|---|---|---|
| `forest.ipynb` | Jupyter notebook | End-to-end pipeline (setup → train → evaluate → export) | README |
| `app.py` | Gradio application | Live Wildlife/Threat inference UI | README, HF Space, `forest.ipynb` |
| `requirements.txt` | Dependency manifest | `pip install` for app and notebook | README |
| `README.md` | Documentation | Project overview, live demo link, setup instructions | — |
| `Neural_Network_Wildlife_Threat_Detection.png` | Image (4.16 MB) | Architecture diagram embedded in README | README (`<img src=...>`) |
| `.gitattributes` | Git config | Git LFS tracking rules | Git |

### 2.2 `dataset/` — data and legacy utilities

| Path | Count / size | Role | Referenced by |
|---|---|---|---|
| `dataset/esc50.csv` | 2,000 rows, 7 columns | ESC-50 metadata | `forest.ipynb` (`pd.read_csv`) |
| `dataset/raw/environmental-sound-classification-50.zip` | 1 file (Git LFS) | Raw ESC-50 archive | `forest.ipynb`, `.gitattributes` |
| `dataset/audio/audio/*.wav` | 2,000 files | Extracted ESC-50 audio (source for curation) | `forest.ipynb` (`audio_dir.glob('*.wav')`) |
| `dataset/curated/wildlife/*.wav` | 200 files | Wildlife class copies | `forest.ipynb` feature extraction |
| `dataset/curated/threat/*.wav` | 160 files | Threat class copies | `forest.ipynb` feature extraction |
| `dataset/processed/X_train.npy` | Git LFS pointer (133 B on disk) | Saved training features | `forest.ipynb` (write only; not loaded back) |
| `dataset/processed/X_val.npy` | Git LFS pointer (132 B on disk) | Saved validation features | `forest.ipynb` (write only) |
| `dataset/processed/y_train.npy` | 2,432 B | Training labels (288 samples) | `forest.ipynb` (write only) |
| `dataset/processed/y_val.npy` | 704 B | Validation labels (72 samples) | `forest.ipynb` (write only) |
| `dataset/processed/norm_stats.npz` | 508 B | `mean`, `std` normalization stats | `forest.ipynb` |
| `dataset/bc_utils.py` | 191 lines | ffmpeg audio utilities, augmentation, BC-learning mix | `utils.py`, `utils2.py`, README |
| `dataset/utils.py` | 341 lines | Thread-safe ESC-50 generator | `bc_utils.py` (imports it); README |
| `dataset/utils2.py` | 262 lines | Keras `Sequence`-based ESC-50 loader | `bc_utils.py`; README |

### 2.3 `models/` — trained artifacts

| Path | Role | Referenced by |
|---|---|---|
| `models/forestguard_cnn.pt` | PyTorch `state_dict` (39 keys) | `forest.ipynb`, README inference example |
| `models/label_map.json` | `{"0": "Wildlife", "1": "Threat"}` | `forest.ipynb`, README |
| `models/norm_stats.npz` | `mean=-43.603737`, `std=18.599327` | `forest.ipynb` (cell 60), README |

### 2.4 External deployment (Hugging Face Space)

| Property | Value |
|---|---|
| **Space page** | https://huggingface.co/spaces/priyansupattanaik/forestguard_CDAC-DNN-Project |
| **Direct app URL** | https://priyansupattanaik-forestguard-cdac-dnn-project.hf.space |
| **SDK** | Gradio 5.34.2 |
| **Entry file** | `app.py` |
| **Space artifacts** | `forestguard_cnn.pt`, `label_map.json`, `norm_stats.npz` (at Space root) |
| **Runtime status** | May show as **PAUSED** on Hugging Face free tier — restart from Space settings |

### 2.5 Files explicitly absent (searched, not found)

- `environment.yml` (mentioned as future work in README)
- `setup.py` / `pyproject.toml`
- Test files (`test_*.py`, `tests/`)
- CI/CD configs (`.github/workflows/`, etc.)
- Docker files

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         forest.ipynb (Active Pipeline)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Setup & Paths ──► Zip Extract ──► EDA ──► Curation ──► Feature Extract    │
│       │                                              │                      │
│       │                                              ▼                      │
│       │                                    Train/Val Split + Norm Stats     │
│       │                                              │                      │
│       │                                              ▼                      │
│       │                              ForestGuardCNN + SpectrogramDataset    │
│       │                                              │                      │
│       │                                              ▼                      │
│       └──────────────────────────────► Train / Eval / Export Artifacts      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              Legacy ESC-50 Loaders (NOT wired into forest.ipynb)            │
├─────────────────────────────────────────────────────────────────────────────┤
│  utils.py ──imports──► bc_utils.py ◄──imports── utils2.py                   │
│  (threadsafe generator)   (ffmpeg, mix, augment)   (Keras Sequence + pathos)│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Architectural characteristics

- **Monolithic notebook design:** All production logic lives in `forest.ipynb`; no modular Python package structure.
- **Offline batch pipeline:** Features are precomputed from WAV files into NumPy arrays before training; no online augmentation during training.
- **CPU/GPU opportunistic:** `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`; last recorded run used CPU.
- **Stateful notebook execution:** Path variables (`ESC50_DIR`, `audio_dir`) are redefined mid-notebook to guard against stale kernel state.

---

## 4. End-to-End Data Flow

### Phase 1 — Environment and path resolution

**Input:** Current working directory (`Path.cwd()`).

**Logic:**
1. If `./dataset` missing, check parent directory.
2. Raise `FileNotFoundError` if still not found.
3. Create directories: `dataset/curated/wildlife`, `dataset/curated/threat`, `dataset/processed`, `models`.

**Output:** Path constants — `BASE_DIR`, `RAW_DIR`, `ESC50_DIR`, `CURATED_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `DEVICE`.

---

### Phase 2 — Raw data extraction

**Input:** `dataset/raw/environmental-sound-classification-50.zip`

**Logic:**
- If `dataset/esc50.csv` exists → skip extraction.
- Else if zip exists → `zipfile.ZipFile.extractall(ESC50_DIR.parent)` (extracts to `BASE_DIR`, i.e., project root).
- Else → `FileNotFoundError`.

**Output:** `dataset/esc50.csv` (verified present; 2,000 samples).

**Edge case:** Extraction target is `BASE_DIR`, not `dataset/`. The notebook assumes post-extraction layout already places `esc50.csv` under `dataset/` and audio under `dataset/audio/audio/`.

---

### Phase 3 — Metadata exploration (EDA 1)

**Input:** `esc50.csv`

**Columns (verified):** `filename`, `fold`, `target`, `category`, `esc10`, `src_file`, `take`

**Verified statistics:**
- Total samples: 2,000
- Fold distribution: 400 samples per fold (folds 1–5)
- ESC-10 subset count: 400 samples (`df['esc10'].sum()`)

**Output:** Pandas DataFrame `df`; seaborn countplot of all 50 categories.

---

### Phase 4 — Binary class curation

**Input:** Full ESC-50 metadata DataFrame.

**Class mapping (hardcoded in notebook):**

```python
WILDLIFE_CLASSES = ["dog", "frog", "crow", "crickets", "chirping_birds"]
THREAT_CLASSES   = ["chainsaw", "siren", "engine", "helicopter"]
```

**Label assignment:**
- `label = 0` if `category ∈ WILDLIFE_CLASSES`
- `label = 1` if `category ∈ THREAT_CLASSES`

**Verified counts from `esc50.csv`:**

| Category | Samples | Mapped label |
|---|---|---|
| dog | 40 | 0 (Wildlife) |
| frog | 40 | 0 |
| crow | 40 | 0 |
| crickets | 40 | 0 |
| chirping_birds | 40 | 0 |
| chainsaw | 40 | 1 (Threat) |
| siren | 40 | 1 |
| engine | 40 | 1 |
| helicopter | 40 | 1 |
| **Total** | **360** | 200 Wildlife / 160 Threat |

**File copy operation:**
- Source: `dataset/audio/audio/{filename}`
- Destination: `dataset/curated/wildlife/` or `dataset/curated/threat/`
- Uses `shutil.copy2` only if destination does not exist
- Missing source files log `⚠️ Source not found: {src}`

**Output:** 200 wildlife WAV + 160 threat WAV in curated folders.

---

### Phase 5 — Feature extraction

**Input:** Curated WAV files.

**Parameters (hardcoded):**

| Parameter | Value | Purpose |
|---|---|---|
| `SR` | 22050 | Resampling rate via `librosa.load` |
| `N_MELS` | 128 | Mel filter banks |
| `FMAX` | 8000 | Maximum frequency for mel bins |
| `TARGET_FRAMES` | 216 | Fixed time dimension |

**`extract_mel_spectrogram(file_path)` algorithm:**
1. `y, sr = librosa.load(file_path, sr=22050)`
2. `mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)`
3. `mel_db = librosa.power_to_db(mel_spec, ref=np.max)`
4. Pad with zeros on time axis if `shape[1] < 216`; truncate to 216 if longer.

**Output:**
- `X`: `np.float32` array shape `(360, 128, 216)`
- `y`: `np.int64` array shape `(360,)`
- Per-class extraction logged with `tqdm` progress bars

**Verified feature statistics (training subset):**
- Min: -80.00 dB
- Max: 0.00 dB
- Mean: -43.60
- Std: 18.60

---

### Phase 6 — Split, normalization, persistence

**Split:**
```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

**Verified split sizes:**
| Array | Shape | Class 0 | Class 1 |
|---|---|---|---|
| `X_train` | (288, 128, 216) | — | — |
| `X_val` | (72, 128, 216) | — | — |
| `y_train` | (288,) | 160 | 128 |
| `y_val` | (72,) | 40 | 32 |

**Normalization stats (computed on `X_train` only):**
- `mean_val = -43.603737`
- `std_val = 18.599327`

**Files written:**
- `dataset/processed/X_train.npy`, `X_val.npy`, `y_train.npy`, `y_val.npy`
- `dataset/processed/norm_stats.npz`
- `models/label_map.json` → `{0: "Wildlife", 1: "Threat"}`
- `models/norm_stats.npz` (duplicate save in final cell)

---

### Phase 7 — Model training and evaluation

See [Section 6](#6-machine-learning-pipeline) for full detail.

**Output artifacts:**
- `models/forestguard_cnn.pt` — best `state_dict` by validation accuracy
- Evaluation metrics on validation set (72 samples)

---

## 5. Module Reference — Inputs, Outputs, and Mechanisms

### 5.1 `forest.ipynb` — active pipeline modules

#### 5.1.1 Path bootstrap (Cell: Setup & Imports)

| | |
|---|---|
| **Inputs** | Filesystem layout (`dataset/` folder) |
| **Outputs** | `BASE_DIR`, `RAW_DIR`, `ESC50_DIR`, `CURATED_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `DEVICE` |
| **Side effects** | Creates curated/processed/models directories |

#### 5.1.2 `extract_mel_spectrogram()`

| | |
|---|---|
| **Input** | `file_path: Path` to a `.wav` file |
| **Output** | `np.ndarray` shape `(128, 216)`, dtype float (dB-scaled mel spectrogram) |
| **Dependencies** | `librosa`, `numpy` |
| **Error handling** | Caller wraps in try/except; logs `⚠️ Error processing {filename}` |

#### 5.1.3 `ForestGuardCNN` (PyTorch `nn.Module`)

| | |
|---|---|
| **Input** | Tensor shape `(batch, 1, 128, 216)` after `SpectrogramDataset` adds channel dim |
| **Output** | Logits shape `(batch, 2)` |
| **Parameters** | 226,442 trainable (notebook count) |

**Architecture (from source code and `print(model)` output):**

```
features (Sequential):
  Block 1: Conv2d(1→24, 3×3, pad=1) → BN → ReLU → Conv2d(24→24) → BN → ReLU → MaxPool2d(2) → Dropout(0.25)
  Block 2: Conv2d(24→48) → BN → ReLU → Conv2d(48→48) → BN → ReLU → MaxPool2d(2) → Dropout(0.35)
  Block 3: Conv2d(48→96) → BN → ReLU → AdaptiveAvgPool2d(4, 4)

classifier (Sequential):
  Flatten → Linear(1536 → 96) → ReLU → Dropout(0.5) → Linear(96 → 2)
```

Where `1536 = 96 × 4 × 4` (output channels × pooled spatial dims).

#### 5.1.4 `SpectrogramDataset` (PyTorch `Dataset`)

| | |
|---|---|
| **Inputs** | `features` (N, 128, 216), `labels` (N,), `mean`, `std` |
| **`__getitem__` output** | `(x, label)` where `x` shape is `(1, 128, 216)` normalized: `(x - mean) / (std + 1e-8)` |
| **dtype** | `float32` features, `long` labels |

#### 5.1.5 `run_epoch(model, loader, criterion, optimizer=None)`

| | |
|---|---|
| **Training mode** | `optimizer is not None` → `model.train()`, backprop enabled |
| **Eval mode** | `optimizer is None` → `model.eval()`, `torch.no_grad()` |
| **Output** | `(avg_loss, accuracy)` where accuracy = `correct / total` |
| **Metrics** | Loss averaged per-sample; accuracy via `argmax` on logits |

---

### 5.2 `dataset/bc_utils.py` — legacy audio utilities

**Not imported by `forest.ipynb`.** Used by `utils.py` and `utils2.py`.

| Function | Inputs | Output | Mechanism |
|---|---|---|---|
| `change_audio_rate(fname, src_dir, rate, dest_dir)` | WAV filename, directories, target Hz | Side effect: writes resampled WAV | Shell `ffmpeg` subprocess if dest file missing |
| `padding(pad)` | Pad width | Callable | `np.pad` constant padding |
| `random_crop(size)` | Crop length | Callable | Random start index slice |
| `normalize(factor)` | Divisor | Callable | Element-wise division |
| `random_scale(max_scale)` | Scale factor | Callable | Linear interpolation time-stretch |
| `random_gain(db)` | dB range | Callable | Random amplitude multiplier |
| `mix(sound1, sound2, r, fs)` | Two signals, mix ratio, sample rate | Mixed signal | A-weighted gain normalization (BC-learning) |
| `compute_gain(sound, fs)` | Signal, sample rate | dB gain array | FFT-based A-weighting or RMSE |
| `noiseAugment(opt)` | Options object with `.data`, `.fs` | Callable | Loads `noise/wav{fs//1000}.npz` (not present in repo) |
| `multi_crop(input_length, n_crops)` | Length, crop count | Callable | Evenly spaced temporal crops |
| `kl_divergence(y, t)` | Tensors | Scalar | References undefined `F` (marked "Wont work with keras") |
| `to_hms(time)` | Seconds | Formatted string | Human-readable duration |

**External dependency:** `ffmpeg` must be on system PATH (documented in README; invoked via `subprocess.call(..., shell=True)`).

---

### 5.3 `dataset/utils.py` — thread-safe ESC-50 generator

**Not imported by `forest.ipynb`.**

| Component | Details |
|---|---|
| **Class `ESC50`** | Yields `(sound, label)` pairs from ESC-50 CSV + WAV directory |
| **Default paths** | `csv_path='../meta/esc50.csv'`, `wav_dir='../audio'` |
| **Default `audio_rate`** | 44100 Hz (via `bc_utils.change_audio_rate`) |
| **`_data_gen()`** | Infinite generator with optional mixing (`U.mix`), strong augmentation |
| **`batch_gen(batch_size)`** | Batched version of `_data_gen` |
| **`get_train_test(test_split)`** | Factory: 4 folds train, 1 fold test |
| **`__main__`** | Calls `test_plot_audio()` using matplotlib |

**Inputs:** ESC-50 CSV, WAV files, fold indices.
**Outputs:** Generator yielding `sound` shape `(samples, 1)` and label (int or one-hot if mixing).

---

### 5.4 `dataset/utils2.py` — Keras Sequence ESC-50 loader

**Not imported by `forest.ipynb`.**

| Component | Details |
|---|---|
| **Class `ESC50(keras.utils.Sequence)`** | Batch-oriented ESC-50 loader |
| **Class `ESC10(ESC50)`** | Filters to ESC-10 subset (10 classes) |
| **`__data_generation`** | Uses `pathos.pools.ProcessPool` (4 workers if mixing, else 1) |
| **Dependencies** | `keras`, `pathos`, `scipy.io.wavfile`, `bc_utils` |
| **`__main__`** | Instantiates `ESC10(mix=True)`, prints `a[0]` timing |

**Inputs:** Same as `utils.py` plus `batch_size`.
**Outputs:** Batched `(X, y)` numpy arrays with categorical labels.

---

## 6. Machine Learning Pipeline

### 6.1 Training configuration (from `forest.ipynb` source)

| Hyperparameter | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Optimizer | `torch.optim.AdamW(lr=5e-4, weight_decay=5e-3)` |
| LR scheduler | `CosineAnnealingLR(T_max=25, eta_min=1e-6)` — stepped every epoch |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping patience | 6 (monitors validation accuracy) |
| Train batches | 18 |
| Val batches | 5 |

### 6.2 Training run results (from notebook cell outputs)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 01 | 0.6253 | 0.6542 | 0.6701 | 0.5972 |
| 10 | 0.3240 | 0.3425 | 0.9375 | 0.9444 |
| 17 | 0.2933 | 0.3121 | 0.9549 | **0.9722** |
| 18–22 | — | — | — | 0.9444–0.9722 |
| 23 | 0.2729 | 0.3080 | 0.9722 | 0.9444 |

- **Early stopping triggered:** epoch 23 (no improvement for 6 epochs after best at epoch 17).
- **Best validation accuracy:** 0.9722 (70/72 correct → 2 misclassifications).

### 6.3 Validation classification report (notebook output)

```
              precision    recall  f1-score   support
    Wildlife       0.97      0.97      0.97        40
      Threat       0.97      0.97      0.97        32
    accuracy                           0.97        72
```

### 6.4 Inference contract (reconstructed from export cells + README)

To run inference on a new audio file, a consumer must replicate the notebook's feature pipeline:

1. Load WAV → resample to 22050 Hz
2. Extract mel spectrogram (128 mels, fmax 8000) → power-to-dB
3. Pad/truncate to 216 frames
4. Apply normalization: `(x - mean) / (std + 1e-8)` using saved stats
5. Add channel dimension → shape `(1, 1, 128, 216)`
6. Forward through `ForestGuardCNN` → `argmax` on 2-class logits
7. Map index via `label_map.json`

**README inference snippet** references `from forest import ForestGuardCNN` — no `forest.py` module exists; the class is defined only inside the notebook.

---

## 7. Technology Stack

### 7.1 Core ML and numerics

| Library | Version | Usage in project | Why it appears here |
|---|---|---|---|
| **PyTorch** (`torch`, `torch.nn`, `torch.utils.data`) | Not pinned | CNN definition, training loop, `DataLoader`, model export | README explicitly chooses PyTorch over Keras for the active pipeline; provides GPU support and `state_dict` serialization |
| **NumPy** | Not pinned | Array storage (`.npy`, `.npz`), feature tensors, normalization | Standard ndarray interchange format for precomputed spectrograms |
| **scikit-learn** | Not pinned | `train_test_split` (stratified), `classification_report`, `confusion_matrix` | Provides stratified 80/20 split and evaluation metrics without custom implementations |
| **librosa** | Not pinned | `load`, `melspectrogram`, `power_to_db`, `display` | Domain-standard audio feature extraction; handles resampling and mel filterbank computation |
| **pandas** | Not pinned | `read_csv`, DataFrame filtering, EDA | ESC-50 metadata is CSV-formatted; pandas enables category-based filtering for curation |

### 7.2 Visualization and notebook UX

| Library | Usage |
|---|---|
| **matplotlib** | Class distribution plots, spectrogram heatmaps, training curves, histograms, confusion matrix |
| **seaborn** | `countplot`, `barplot`, `heatmap` with `whitegrid` theme |
| **tqdm** | Progress bars for file copying and feature extraction |
| **IPython.display** | Inline audio playback of curated samples |

### 7.3 Legacy stack (utils.py / utils2.py only)

| Library | Usage | Note |
|---|---|---|
| **scipy.io.wavfile** | Raw WAV I/O in legacy loaders | Bypassed by librosa in active pipeline |
| **keras** | `utils.Sequence`, `to_categorical` | Only in `utils2.py`; not used by notebook |
| **pathos** | Multiprocessing pool | Only in `utils2.py`; `ModuleNotFoundError` if imported without install |
| **ffmpeg** (system) | Audio resampling in `bc_utils.py` | Not used by `forest.ipynb` (librosa resamples at load time) |
| **threading** | `threadsafe_generator` decorator | Only in `utils.py` |

### 7.4 Infrastructure tools

| Tool | Configuration | Purpose |
|---|---|---|
| **Git LFS** | `.gitattributes` | Tracks large binaries: `X_train.npy`, `X_val.npy`, raw zip |
| **Jupyter** | `forest.ipynb` kernelspec: Python 3 | Interactive development and execution environment |

### 7.5 Installation instructions (from README — not locked to versions)

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn librosa jupyter
```

Additional undeclared dependencies used by the notebook: `tqdm`.

Additional undeclared dependencies for legacy modules: `scipy`, `keras`, `pathos`.

---

## 8. Configuration and Infrastructure

### 8.1 `.gitattributes` (Git LFS rules)

```
/dataset/processed/X_train.npy  → filter=lfs
/dataset/processed/X_val.npy  → filter=lfs
/dataset/raw/environmental-sound-classification-50.zip → filter=lfs
```

### 8.2 Hardcoded constants (no external config files)

All pipeline parameters are embedded directly in `forest.ipynb` cells. There is no `.env`, `config.yaml`, or CLI argument parsing.

| Constant | Location | Value |
|---|---|---|
| `WILDLIFE_CLASSES` / `THREAT_CLASSES` | Curation cell | 5 + 4 category names |
| `SR`, `N_MELS`, `FMAX`, `TARGET_FRAMES` | Feature extraction cell | 22050, 128, 8000, 216 |
| `BATCH_SIZE` | Training setup | 16 |
| `EPOCHS`, `PATIENCE` | Training loop | 30, 6 |
| `random_state` | `train_test_split` | 42 |

### 8.3 ESC-50 fold structure

ESC-50 provides 5 predefined folds (400 samples each). The active pipeline does **not** use fold-based cross-validation; it uses a random stratified split on the 360-sample curated subset. The legacy `utils.py` `get_train_test()` **does** use fold-based splitting (4 train folds, 1 test fold).

---

## 9. Runtime Artifacts and On-Disk State

### 9.1 Checkpoint tensor inventory (`forestguard_cnn.pt`)

39 state dict keys including:
- Conv weights/biases for 6 `Conv2d` layers (1→24→24→48→48→96)
- BatchNorm running stats for 5 `BatchNorm2d` layers
- Linear layers: `(1536 → 96)` and `(96 → 2)`

### 9.2 Normalization statistics (verified identical in both locations)

| File | mean | std |
|---|---|---|
| `models/norm_stats.npz` | -43.603737 | 18.599327 |
| `dataset/processed/norm_stats.npz` | -43.603737 | 18.599327 |

### 9.3 Label map (`models/label_map.json`)

```json
{"0": "Wildlife", "1": "Threat"}
```

Note: JSON keys are strings (`"0"`, `"1"`) after export; written from Python dict with integer keys `{0: "Wildlife", 1: "Threat"}`.

### 9.4 Git LFS pointer state (local workspace)

`X_train.npy` and `X_val.npy` on disk are 133/132-byte LFS pointer files, not actual feature arrays. Running `git lfs pull` or re-executing the feature extraction cell is required to materialize them.

---

## 10. Known Gaps, Defects, and Documentation Drift

| Issue | Evidence | Impact |
|---|---|---|
| **`import copy` missing** | Training cell uses `copy.deepcopy(model.state_dict())` but setup cell has no `import copy` | `NameError` on fresh top-to-bottom notebook execution |
| **`forest.py` referenced historically** | Older README inference example used `from forest import ForestGuardCNN` | Class now available via `app.py` |
| **HF Space may be paused** | Hugging Face API reports `runtime.stage: PAUSED` | Live demo URL requires Space restart on HF |
| **README architecture drift** | README claims FC layer with 128 units; code uses `Linear(1536, 96)` | Documentation inaccuracy |
| **README early stopping epoch** | README: epoch 18; notebook output: epoch 23 | Documentation inaccuracy |
| **README conv block 3** | README lists MaxPool + Dropout in block 3; code uses `AdaptiveAvgPool2d(4,4)` with no third MaxPool/Dropout | Documentation inaccuracy |
| **No `requirements.txt`** | README lists pip commands but no lockfile | Non-reproducible environments |
| **No test suite** | No test files found | No automated regression protection |
| **Legacy utils unused** | Zero imports from `forest.ipynb` to `utils.py`/`utils2.py`/`bc_utils.py` | Dead code path unless manually invoked |
| **`noiseAugment` data missing** | `bc_utils.noiseAugment` expects `noise/wav*.npz` not in repo | Function fails if called |
| **`kl_divergence` broken** | References undefined `F` (PyTorch functional) | Function non-functional as written |
| **Class imbalance** | 200 Wildlife vs 160 Threat (1.25:1 ratio) | Mitigated partially by stratified split but not by weighted loss |
| **"dog" as Wildlife** | `dog` in `WILDLIFE_CLASSES` | Design choice that may cause domain confusion in deployment |

---

## 11. Technical Interview Questions

Questions below are grounded in this specific codebase. They probe design decisions, scalability limits, and implementation details a senior engineer would raise.

### Architecture and design

1. **Why does the active pipeline use librosa at 22,050 Hz while the legacy `bc_utils.py` utilities default to 44,100 Hz and shell out to ffmpeg?** What are the trade-offs of consolidating on one resampling path?

2. **The notebook references `app.py` for deployment parity, but no such file exists. How would you extract `ForestGuardCNN`, `extract_mel_spectrogram`, and `SpectrogramDataset` into an importable inference module without breaking the saved checkpoint?**

3. **Why was a custom CNN chosen over a pretrained audio backbone (e.g., PANNs, YAMNet, AST)?** Given only 360 training samples, what evidence in the results supports or challenges this choice?

4. **`ForestGuardCNN` uses `AdaptiveAvgPool2d(4, 4)` instead of a fixed MaxPool in the final conv block. How does this affect the model's sensitivity to input time dimension changes? What happens if `TARGET_FRAMES` is modified without retraining?**

### Data pipeline

5. **The curation step maps `dog` to Wildlife and uses only 9 of 50 ESC-50 categories. What domain assumptions does this encode, and how would you defend or challenge this mapping in a conservation monitoring context?**

6. **Why does the pipeline copy 360 files into `dataset/curated/` instead of training directly from `dataset/audio/audio/` with a manifest CSV?** What are the storage, reproducibility, and synchronization implications?

7. **ESC-50 provides 5-fold cross-validation splits, but the notebook uses `train_test_split` on the curated subset. When would fold-based evaluation be more appropriate, and what risk does the current approach introduce?**

8. **Feature extraction is fully offline; no augmentation is applied during training. Given the small dataset (288 training samples), what augmentations would be compatible with the existing mel-spectrogram pipeline, and where would you insert them?**

### Machine learning

9. **The model uses `CrossEntropyLoss(label_smoothing=0.1)`. With only 72 validation samples, how does label smoothing interact with the reported 97.22% accuracy, and could it mask overconfidence on the minority Threat class?**

10. **Training employs AdamW (`weight_decay=5e-3`) plus dropout (0.25, 0.35, 0.5) plus early stopping (patience=6). Which of these regularizers contributed most given the train/val accuracy gap at epoch 17 (97.22% val vs 95.49% train)?**

11. **Why is normalization computed as global mean/std over all mel bins and time frames rather than per-sample or per-frequency-bin normalization?** How would this choice affect deployment on out-of-distribution forest recordings?

12. **The saved checkpoint has 226,442 trainable parameters for a 2-class problem with 288 training samples. What is the parameter-to-sample ratio, and what overfitting signals do you see in the epoch-by-epoch log?**

### Engineering and operations

13. **The training cell calls `copy.deepcopy` without `import copy`. How would you harden this notebook for CI execution (e.g., `nbconvert --execute`), and what other latent runtime defects would you scan for?**

14. **`X_train.npy` and `X_val.npy` are Git LFS pointers in the local workspace. Describe the failure modes when a new developer clones the repo without `git lfs pull` and attempts to train from saved arrays.**

15. **There is no `requirements.txt`. How would you construct a reproducible environment specification from the actual imports across `forest.ipynb`, `utils.py`, and `utils2.py`?**

16. **The legacy `utils2.py` uses `pathos.ProcessPool` for parallel sample generation, while the active pipeline uses single-threaded `tqdm` loops. At what dataset scale does the multiprocessing approach become worthwhile, and what are the pickling constraints on Windows?**

### Scalability and production

17. **If this system were deployed for real-time forest monitoring with streaming audio, which stages of the current batch pipeline become bottlenecks?** Quantify the cost of mel-spectrogram extraction relative to CNN inference.

18. **How would you design a model serving API that accepts arbitrary-length WAV files, given the fixed `TARGET_FRAMES=216` constraint?** Discuss padding, sliding windows, and aggregation strategies.

19. **The binary classifier conflates four threat categories (chainsaw, siren, engine, helicopter) into one label. What multi-class or hierarchical alternatives would improve operator trust, and what labeled data would you need?**

20. **Given class counts of 200 Wildlife / 160 Threat and a stratified 80/20 split yielding 160/128 train and 40/32 val, how would you implement weighted sampling or focal loss, and what metric would you optimize instead of raw accuracy?**

### Testing and validation

21. **How would you write an integration test that verifies the full path from a single curated WAV file to a correct tensor shape `(1, 128, 216)` and a valid model forward pass, without depending on the full 360-sample dataset?**

22. **The confusion matrix shows 2 errors out of 72 validation samples. What additional analysis (per-source-category breakdown, error audio inspection, confidence calibration) would you perform before deploying to a production forest monitoring system?**

23. **How do you detect spectral distribution shift between ESC-50 studio recordings and field-deployed microphone data, using the saved `norm_stats.npz` as a reference?**

### Security and maintainability

24. **`bc_utils.change_audio_rate` constructs ffmpeg commands via `subprocess.call(cmd, shell=True)`. What are the security and portability implications, and how would you rewrite this for production?**

25. **The project has three overlapping data loading implementations (`forest.ipynb`, `utils.py`, `utils2.py`). Propose a consolidation plan that preserves backward compatibility for ESC-50 fold-based training while maintaining the PyTorch mel-spectrogram pipeline.**

---

## Appendix A — Verified Import Graph

```
forest.ipynb
  ├── numpy, pandas, matplotlib, seaborn, tqdm
  ├── torch, torch.nn, torch.utils.data
  ├── sklearn.model_selection, sklearn.metrics
  ├── librosa, librosa.display
  ├── IPython.display (optional audio cell)
  └── pathlib, os, json, shutil, zipfile, random

utils.py → bc_utils.py
utils2.py → bc_utils.py, keras, pathos, scipy.io.wavfile

No imports from forest.ipynb → utils.py | utils2.py | bc_utils.py
```

## Appendix B — File Count Summary (post-audit workspace)

| Extension | Count |
|---|---|
| `.wav` | 2,360 |
| `.py` | 3 |
| `.npy` | 4 |
| `.npz` | 2 |
| `.json` | 1 |
| `.pt` | 1 |
| `.ipynb` | 1 |
| `.zip` | 1 |
| `.csv` | 1 |
| `.md` | 2 (README + this document) |
| `.png` | 1 |
| `.gitattributes` | 1 |

---

*Document generated from codebase audit. All metrics, paths, and code references reflect on-disk evidence in `forestguard/`.*