# Quick Start

## 1) Activate environment (Windows PowerShell)

```powershell
cd C:\Users\prakh\Desktop\NLU\new\NLU_Assignment_1_Fresh
.\venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3) Prepare dataset

```powershell
python src/prepare_dataset.py
```

Generated files:
- `data/processed/complete_dataset.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/dataset_summary.csv`

## 4) Train and evaluate models

```powershell
python src/train_models.py
```

Generated files:
- `results/all_results.csv`
- `results/per_class_results.csv`
- `results/confusion_matrices.csv`
- `models/*.joblib`

## 5) Open report and GitHub page content
- Detailed report: `docs/report.md`
- GitHub Pages content: `site/index.md`

## 6) Publish GitHub Pages
1. Push this project to your own GitHub repository.
2. In GitHub → Settings → Pages:
   - Source: Deploy from a branch
   - Branch: `main`
   - Folder: `/site`
3. Save and wait for deployment.
