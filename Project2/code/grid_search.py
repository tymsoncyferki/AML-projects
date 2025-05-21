""" module containing functions useful for running grid search """

from utils import *
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def custom_scorer(estimator, X, y):
    """ scorer for grid search """
    y_prob = estimator.predict_proba(X)[:, 1]
    return custom_score(y, y_prob, num_features=X.shape[1], top_k_reference=1000)

def plot_distribution(y_pred):
    """ plots distribution of predicted probabilities with thresholds (0.5 and top 1000) """
    y_pred_sorted = np.sort(y_pred)
    top_1000_threshold = y_pred_sorted[-1000]
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_sorted, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.axvline(x=top_1000_threshold, color='green', linestyle='--', label='Top 1000 threshold')
    
    plt.title('Probability distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count (bin)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def metrics_top_and_global(y_true, y_prob, num_features=2, top_k=1000):
    """ calculates all metrics and returns a dict """
    score= custom_score(y_true, y_prob, num_features=num_features, top_k_reference=top_k)
    logloss = log_loss(y_true, y_prob)
    
    top_k_idx = np.argsort(y_prob)[-top_k:]
    y_true_top = y_true[top_k_idx]
    y_pred_top = (y_prob[top_k_idx] >= 0.5).astype(int)
    acc_top = np.mean(y_true_top == y_pred_top)

    y_pred_all = (y_prob >= 0.5).astype(int)
    acc_global = np.mean(y_true == y_pred_all)
    logloss_top = log_loss(y_true[top_k_idx], y_prob[top_k_idx])

    return {
        "score": score,
        "acc_top": acc_top,
        "acc_global": acc_global,
        "logloss_top": logloss_top,
        "logloss_global": logloss
    }

def evaluate_model_with_gridsearch(model, param_grid, X, y, X_test, n_splits=5, num_features=2, top_k=1000):
    """ runs cross-validation grid search, uses custom_scorer as scorer """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"clf__" + k: v for k, v in param_grid.items()},
        cv=skf,
        scoring=custom_scorer,
        n_jobs=-1
    )

    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_prob_all = cross_val_predict(best_model, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics = metrics_top_and_global(y, y_prob_all, num_features, top_k)
    y_pred = best_model.predict_proba(X_test)[:, 1]
    plot_distribution(y_pred)
    
    return best_model, grid.best_score_, grid.best_params_, metrics

""" model parameters grid """
models_params = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        }
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    ),
    "SVC": (
        SVC(probability=True),
        {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    ),
    "Naive Bayes": (
        GaussianNB(),
        {}
    ),
    "AdaBoost": (
        AdaBoostClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    )
}
