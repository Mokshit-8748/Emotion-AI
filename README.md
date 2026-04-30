# Emotion AI

Speech emotion recognition project with two main user-facing paths:

- `Fine-Tuned WavLM` for the strongest production inference and evaluation flow
- `Original Model` as the legacy handcrafted-feature baseline

## Project Layout

- `dashboard.py`: Streamlit dashboard entrypoint
- `app_pages/`: dashboard pages
- `src/`: active shared Python modules
- `models/`: active checkpoints used by the dashboard
- `legacy/`: quarantined older experiments and superseded scripts

## Setup

This project is designed to run from the project root:

```powershell
cd "C:\Users\91970\Desktop\New folder\Emotion AI"
venv\Scripts\activate
```

If you need to install dependencies into a fresh environment:

```powershell
pip install -r requirements.txt
```

## Run The Dashboard

```powershell
python -m streamlit run dashboard.py
```

## Main Training Commands

Fine-Tuned WavLM:

```powershell
python train_finetune_3dataset.py --device cuda --datasets data\RAVDESS,data\CREMA-D,data\TESS --output models\emotion_wavlm_finetuned_3dataset_run.pt --metadata models\emotion_wavlm_finetuned_3dataset_run.json
```

Original baseline:

```powershell
python src\train_model.py --datasets data\RAVDESS,data\CREMA-D,data\TESS --epochs 120 --batch_size 128 --lr 3e-4 --dropout 0.45 --output models\emotion_ser_model.h5
```

GAN-enhanced baseline:

```powershell
python train_gan_model.py
```

## Notes

- Large datasets and model artifacts are intentionally ignored by Git.
- The current best dashboard checkpoint is `models/emotion_wavlm_finetuned_3dataset_v2.pt`.
- Older experimental code has been moved into `legacy/` to keep the active project surface smaller.

