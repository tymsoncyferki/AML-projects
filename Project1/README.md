# Project 1 - CyclicCD

## Structure
### Files:
- logreg.py - file with implementation of CCD algorithm, includes function for generating synthetic dataset
- datasets.ipynb - notebook for downloading and preparing datasets
- experiments.ipynb - notebook for running experiments on real datasets
- experiments_synthetic.ipynb - notebook for running experiments on synthetic dataset
- LogRegCCDDemo.ipynb - demo usage of CCD algorithm
- requirements.txt - venv requirements for running the project

### Folders:
- /raw_data - original data files
- /data - preprocessed data files
- /results - folder with result plots and tables

## Logistic Regression CCD Model Guide

### Prerequisites

Before running the model, ensure you have the necessary libraries installed:

```bash
pip install -r requirements.txt
```

### 1. Fit Model

Initialize and train the logistic regression model:

```python
from logreg import *
ccd_model = LogRegCCD()
results, best_lambda = ccd_model.optimize_lambda(X_train, y_train, X_valid, y_valid)
ccd_model.fit(X_train, y_train, best_lambda)
```

### 2. Predict

Make predictions on the validation set:

```python
ccd_probs = ccd_model.predict_proba(X_valid)
ccd_preds = (ccd_probs >= 0.5).astype(int)
```

### 3. Evaluate

Calculate and print the F1 score:

```python
ccd_f1 = f1_score(y_valid, ccd_preds)
print(f"LogRegCCD F1 Score: {ccd_f1:.4f}")
```

### 4. Visualize Coefficients

Plot the coefficient paths:

```python
ccd_model.plot_coefficients(results)
```

### 5. Visualize Performance

Plot the F1 score against lambda values:

```python
ccd_model.plot(results, measure="f1")
```
