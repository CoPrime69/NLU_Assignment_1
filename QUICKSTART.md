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
- `results/cv_results.csv`
- `results/full_data_models.csv`
- `models/*.pkl`

## 5) Generate visualizations

```powershell
python src/generate_visualizations.py
```

Generated files:
- `results/visualizations/*.png`

## 6) Open report and GitHub page content
- Detailed report: `docs/report.md`
- GitHub Pages content: `README.md` (or `index.md` if you add one)

## 7) Publish GitHub Pages
1. Push this project to your own GitHub repository.
2. In GitHub → Settings → Pages:
   - Source: Deploy from a branch
   - Branch: `main`
   - Folder: `/ (root)`
3. Save and wait for deployment.
