# VitalAI Dataset Directory

Place the Kaggle **"Disease Prediction Using Machine Learning"** CSV files in this directory:

- `data/Training.csv`
- `data/Testing.csv`

## Expected Schema

- Binary symptom columns (`0` or `1`) representing patient symptoms (132 columns in the standard Kaggle dataset).
- Target column named `prognosis` containing the disease label (41 classes in the standard Kaggle dataset).
- Note: `backend/ml/data_loader.py` automatically removes trailing empty `Unnamed:*` columns and normalizes extra whitespace in column names and disease labels.

## Run Model Training & Anti-Overfitting Evaluation

```powershell
python -m backend.ml.train_model --data-dir data
python -m backend.ml.evaluate_model --data-dir data
```
