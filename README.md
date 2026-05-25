# CrysText — Text-Conditioned Crystal Structure Generation

Generate valid crystal structure CIF files from a material formula and space group number using a fine-tuned large language model.

---

## What It Does

Type a formula like `NaCl` and space group `225` → CrysText generates a complete Crystallographic Information File (CIF) with cell parameters, atom positions, and symmetry information, rendered as an interactive 3D ball-and-stick structure.

---

## Model

- **Base:** Mistral-7B-v0.3
- **Fine-tuning:** QLoRA (r=16, lora_alpha=16), 4-bit quantization
- **Dataset:** MP-20 (27,136 experimentally verified crystal structures)
- **Training:** Session 3 complete (10k samples, loss=0.2466) — Sessions 4 & 5 pending
- **HuggingFace:** https://huggingface.co/rakshitha9/crystext-mistral-10k

---

## Requirements

- Python 3.10+
- CUDA GPU with at least **8GB VRAM**
- CUDA drivers installed

---

## Installation

```bash
git clone https://github.com/YOURUSERNAME/crystext.git
cd crystext
pip install -r requirements.txt
```

---

## Running The App

You need two terminals open at the same time.

**Terminal 1 — Start the Flask backend (loads the AI model):**
```bash
python app.py
```
Wait until you see:
```
✅ CrysText is ready! Server starting on http://localhost:5000
```
This takes 10-15 minutes on first run — the model is downloading and loading.

**Terminal 2 — Start the frontend:**
```bash
python -m http.server 8080
```

**Then open your browser and go to:**
```
http://localhost:8080
```

---

## Good Demo Compounds

All of these are in MP-20 and generate well:

| Formula | Space Group | Structure Type |
|---|---|---|
| NaCl | 225 | Rock salt (cubic) |
| GaAs | 216 | Zinc blende |
| BaTiO3 | 99 | Perovskite |
| TiO2 | 136 | Rutile |
| Fe2O3 | 167 | Hematite |
| MgO | 225 | Rock salt (cubic) |

---

## Project Structure

```
crystext/
├── app.py                        ← Flask backend (port 5000)
├── index.html                    ← Frontend with 3D viewer (port 8080)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── HOW_TO_TRAIN.md               ← Instructions for teammates to run training
└── training/
    └── crystext_training.ipynb   ← Kaggle training notebook
```

---

## How It Works

1. User enters formula + space group in the frontend
2. Frontend sends POST request to Flask on port 5000
3. Flask formats it into Alpaca instruction prompt
4. Mistral-7B generates CIF text token by token (greedy decoding)
5. Pymatgen validates the CIF and converts primitive cell to conventional cell
6. Validated CIF is returned to frontend
7. Three.js renders the 3D ball-and-stick structure

---

## Known Limitations

- Trained on 10k/27k samples — space group conditioning improves with Sessions 4 & 5
- Works best for compounds present in MP-20
- Generation takes 30-60 seconds per structure

---

## Team

Built by a team of 5 as part of an AI + Materials Science project.
Training done on Kaggle T4 GPU. Inference runs locally on RTX 5050.
