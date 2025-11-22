
# GANformer Battlefield Model (GAT) — Local Setup Guide

This repository explains how to:

1. Install a Python virtual environment on your own machine.
2. Clone and use the **official GANformer (GAT)** repository by Dor Arad.
3. Download and use our **pretrained battlefield GANformer model**.
4. Train GANformer on **your own dataset** and generate images.

All model code comes from the original GANformer repository:  
👉 https://github.com/dorarad/gansformer

This repo only provides:

- A **pretrained battlefield checkpoint** (`network-snapshot-001224.pkl`).
- Clear instructions to set everything up **locally** (no Colab).

---

## 1. Requirements

You will need:

- Python **3.8–3.10**
- Git
- An NVIDIA GPU with CUDA (for training / fast generation)
- `pip` (Python package manager)

Recommended OS: **Linux**.  
Windows with WSL2 or native Python can also work if you are comfortable with it.

> ⚠️ Note: GANformer uses custom CUDA ops.  
> Make sure your **PyTorch + CUDA** setup is consistent.  
> If the *original* GANformer repo runs (e.g. their bedrooms example), this battlefield model will also run.

---

## 2. Clone the official GANformer repository

Open a terminal and run:

```bash
git clone https://github.com/dorarad/gansformer.git
cd gansformer/pytorch_version
```

You should now see files such as:

- run_network.py
- generate.py
- prepare_data.py
- torch_utils/ etc.

All commands below assume you are in this folder:

```bash
cd gansformer/pytorch_version
```

---

## 3. Create and activate a virtual environment

### 3.1 Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your shell prompt.

### 3.2 Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, you may need to adjust the execution policy or use `cmd.exe` and `.\.venv\Scriptsctivate.bat`.

---

## 4. Install Python dependencies

GANformer was originally developed for older PyTorch & CUDA versions. The exact combination depends on your system.

### Option A — Use your existing working setup

If you already have **PyTorch + CUDA** installed and working, just install the extra libraries:

```bash
pip install numpy pillow scipy h5py opencv-python
```

Then later test by running a `generate.py` command.  
If it runs without errors, you are good.

### Option B — Follow the original GANformer guidance

Read the original GANformer README / issues for the recommended PyTorch + CUDA versions.

As a rough example (this is **not** guaranteed):

```bash
pip install "torch==1.10.0" "torchvision==0.11.1"  # Match this to your CUDA
pip install numpy pillow scipy h5py opencv-python
```

If you see errors about `upfirdn2d_plugin`, `conv2d_gradfix`, or `custom_ops`, it usually means there is a mismatch between:

- your PyTorch / CUDA version  
and
- the compiled custom CUDA ops.

Fixing that is system-specific and beyond this README; the important point is:  
👉 if the official `gansformer` repo runs for you, the battlefield model here will also run.

---

## 5. Download the pretrained **battlefield** model

We provide a snapshot trained on a **battlefield scene dataset** at 256×256 resolution, stopped at **1224 kimg**.

Download it via GitHub Releases:

```bash
# From inside gansformer/pytorch_version
wget https://github.com/mag-and-krotons/GAT/releases/download/v1.0.0/network-snapshot-001224.pkl
```

- Replace `YOUR_USERNAME/YOUR_REPO` and the tag (`v1.0.0`) with your actual values.
- You get this URL from: **Repo → Releases → your release → right-click the .pkl asset → "Copy link address"**.

After downloading, you should have:

gansformer/pytorch_version/battlefield_snapshot_001224.pkl

---

## 6. Generate images from the battlefield model

With your virtual environment activated and the model downloaded:

```bash
cd gansformer/pytorch_version
source .venv/bin/activate      # or .\.venv\Scriptsctivate on Windows

python generate.py   --gpus 0   --model battlefield_snapshot_001224.pkl   --images-num 36   --output-dir battlefield_samples   --truncation-psi 0.7
```

Explanation of arguments:

- `--gpus 0` — use GPU 0.
- `--model` — path to the checkpoint .pkl file.
- `--images-num` — how many images to generate.
- `--output-dir` — where the PNG files will be saved.
- `--truncation-psi` — controls diversity vs. quality:
  - lower (e.g. 0.5–0.7): fewer weird images, less diversity;
  - higher (e.g. 0.8–1.0): more diverse, but also more artifacts.

Once it finishes, inspect the output directory:

```bash
ls battlefield_samples
```

You should see files like 000000.png, 000001.png, etc.

---

## 7. (Optional) Continue training the battlefield model

If you want to **continue training** our battlefield model on your own machine:

1. You must have prepared a matching dataset as `datasets/battlefield/`.
2. Then you can start a new experiment initialized from the pretrained snapshot:

```bash
cd gansformer/pytorch_version
source .venv/bin/activate

python run_network.py   --train   --gpus 0   --ganformer-default   --expname battlefield-finetune   --dataset battlefield   --data-dir datasets   --resolution 256   --pretrained-pkl battlefield_snapshot_001224.pkl   --eval-images-num 5000   --metrics fid
```

- `--pretrained-pkl battlefield_snapshot_001224.pkl` loads our weights.
- `--expname battlefield-finetune` writes logs and snapshots to:

  results/battlefield-finetune-000/

You can stop training whenever you like; checkpoints are saved periodically.

---

## 8. Train GANformer on **your own dataset**

You can also use the original GANformer code to train on **any** image dataset.

### 8.1 Prepare your images

Organize your images like this:

gansformer/pytorch_version/data_raw/mydataset_raw/
    img0001.jpg
    img0002.jpg
    ...

Optionally resize them to 256×256:

```bash
cd gansformer/pytorch_version
source .venv/bin/activate

python - << 'EOF'
from pathlib import Path
from PIL import Image

src = Path("data_raw/mydataset_raw")
dst = Path("data_raw/mydataset_256")
dst.mkdir(parents=True, exist_ok=True)

count = 0
for p in src.glob("*"):
    if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    img = Image.open(p).convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(dst / (p.stem + ".jpg"), quality=95)
    count += 1

print(f"Resized {count} images into {dst}")
EOF
```

Now you have a clean 256×256 version of your dataset at:

data_raw/mydataset_256/

### 8.2 Build a GANformer dataset

Convert this into the internal GANformer format:

```bash
python prepare_data.py   --task mydataset   --images-dir data_raw/mydataset_256   --format jpg   --ratio 1.0   --data-dir datasets
```

This will create:

datasets/mydataset/

Here, `mydataset` is the **dataset name** you’ll use with `--dataset`.

### 8.3 Train from scratch on your dataset

Start training:

```bash
python run_network.py   --train   --gpus 0   --ganformer-default   --expname mydataset-scratch   --dataset mydataset   --data-dir datasets   --resolution 256   --total-kimg 1000   --eval-images-num 5000   --metrics fid
```

- `--expname mydataset-scratch` → results go into:

  results/mydataset-scratch-000/

- `--total-kimg` controls how long to train:
  - 200–500 kimg: quick experiment.
  - 1000–2000+ kimg: more serious run (longer time).

Inside results/mydataset-scratch-000/ you’ll find:

- network-snapshot-000XYZ.pkl — generator checkpoints.
- fakes000XYZ.png — sample grids.
- metric-fid*.txt — if FID was enabled.

---

## 9. Generate images from your own trained model

Once you have a snapshot you like (e.g. `network-snapshot-000300.pkl`):

```bash
cd gansformer/pytorch_version
source .venv/bin/activate

python generate.py   --gpus 0   --model results/mydataset-scratch-000/network-snapshot-000300.pkl   --images-num 36   --output-dir samples_mydataset_000300   --truncation-psi 0.7
```

Check the output:

```bash
ls samples_mydataset_000300
```

You should see your generated images there.

---

## 10. Summary

- **Code**: we strictly use the original GANformer implementation from  
  https://github.com/dorarad/gansformer
- **This repo**: provides a *battlefield-trained* checkpoint (`network-snapshot-001224.pkl`) and a clear way to:
  - set up a local virtual environment,
  - download the checkpoint,
  - generate images,
  - and train GANformer on your own dataset.

If you can successfully run the official GANformer repo on your machine,  
you can plug in this battlefield model and your own images using the commands above.
