import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, 
    average_precision_score
)
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import bernoulli, multivariate_normal
from tqdm import tqdm

class LogRegCCD:
    """
    Logistic Regression with Coordinate Descent for Elastic Net Regularization.

    Attributes:
        lambda_min (float): Minimum value of regularization strength (lambda).
        lambda_max (float): Maximum value of regularization strength (lambda).
        num_lambdas (int): Number of lambda values to evaluate during optimization.
        alpha (float): Elastic net mixing parameter. alpha=1 corresponds to LASSO (L1),
                      while alpha=0 corresponds to Ridge (L2).
        coefficients (np.ndarray): Coefficients of the logistic regression model, including intercept.
        lambdas (np.ndarray): Array of lambda values to evaluate.
        best_lambda (float): The lambda value that achieved the best validation score.
    """

    def __init__(self, lambda_min=1e-3, lambda_max=1.0, num_lambdas=100, alpha=0.2):
        """
        Initializes the LogRegCCD class with specified hyperparameters.

        Args:
            lambda_min (float): Minimum value of lambda for regularization. Default is 1e-3.
            lambda_max (float): Maximum value of lambda for regularization. Default is 1.0.
            num_lambdas (int): Number of lambda values to evaluate. Default is 100.
            alpha (float): Mixing parameter for elastic net. Default is 0.2.
        """
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_lambdas = num_lambdas
        self.alpha = alpha
        self.coefficients = None
        self.lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num_lambdas)
        self.best_lambda = None

    def _sigmoid(self, z):
        """
        Sigmoid function to calculate the probability of a binary classification.

        Args:
            z (float or np.ndarray): Input value(s).

        Returns:
            np.ndarray: Sigmoid-transformed probabilities.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, y_train, lmbda=None):
        """
        Fits the logistic regression model using coordinate descent.

        Args:
            X_train (np.ndarray): Feature matrix for training data.
            y_train (np.ndarray): Target vector for training data.
            lmbda (float, optional): Regularization strength. Defaults to lambda_min.

        Returns:
            None
        """
        if lmbda is None:
            lmbda = self.lambda_min

        n_samples, n_features = X_train.shape
        self.coefficients = np.random.normal(0, 0.01, n_features + 1)

        for _ in range(100):
            intercept = self.coefficients[0]
            weights = self.coefficients[1:]

            for j in range(n_features):
                partial_residual = y_train - self._sigmoid(
                    np.dot(np.delete(X_train, j, axis=1), np.delete(weights, j)) + intercept)
                gradient = np.dot(X_train[:, j].T, partial_residual.T) / n_samples
                l1_penalty = self.alpha * lmbda
                l2_penalty = (1 - self.alpha) * lmbda
                soft_threshold = np.sign(gradient) * max(0.0, abs(gradient) - l1_penalty)
                weights[j] = soft_threshold.item() / (1 + l2_penalty)

            intercept += np.mean(y_train - self._sigmoid(np.dot(X_train, weights) + intercept))
            self.coefficients[0] = intercept
            self.coefficients[1:] = weights
            log_loss = -np.mean(y_train * np.log(np.asarray(self._sigmoid(np.dot(X_train, self.coefficients[1:]) + self.coefficients[0]).T).reshape(-1)) +
                                (1 - y_train) * np.log(1 - np.asarray(self._sigmoid(np.dot(X_train, self.coefficients[1:]) + self.coefficients[0]).T).reshape(-1)))
            print(f"Iteration {_}, Log-loss: {log_loss}")

    def validate(self, X_valid, y_valid, measure="f1"):
        """
        Validates the model on a validation dataset using the specified performance measure.

        Args:
            X_valid (np.ndarray): Feature matrix for validation data.
            y_valid (np.ndarray): Target vector for validation data.
            measure (str): Performance metric to use. Supported options are:
                           "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc".

        Returns:
            float: The calculated performance score.

        Raises:
            ValueError: If an unsupported measure is provided.
        """
        probabilities = self.predict_proba(X_valid)
        predictions = (probabilities >= 0.5).astype(int)

        if measure == "precision":
            return precision_score(y_valid, predictions)
        elif measure == "recall":
            return recall_score(y_valid, predictions)
        elif measure == "f1":
            return f1_score(y_valid, predictions)
        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_valid, predictions)
        elif measure == "roc_auc":
            return roc_auc_score(y_valid, probabilities)
        elif measure == "pr_auc":
            return average_precision_score(y_valid, probabilities)
        else:
            raise ValueError("Unsupported measure: {}".format(measure))

    def predict_proba(self, X_test):
        """
        Predicts probabilities for the test dataset.

        Args:
            X_test (np.ndarray): Feature matrix for test data.

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        return np.asarray(self._sigmoid(np.dot(X_test, self.coefficients[1:]) + self.coefficients[0]).T).reshape(-1)

    def optimize_lambda(self, X_train, y_train, X_valid, y_valid, measure="f1", verbose=0):
        """
        Optimizes the lambda value by evaluating performance on the validation set.

        Args:
            X_train (np.ndarray): Feature matrix for training data.
            y_train (np.ndarray): Target vector for training data.
            X_valid (np.ndarray): Feature matrix for validation data.
            y_valid (np.ndarray): Target vector for validation data.
            measure (str): Performance metric to optimize. Default is "f1".
            verbose (int): Log level. 0 for no printing, 1 for progress bar, 2 for detailed info.

        Returns:
            dict: A dictionary containing lambda values, performance scores, and coefficient snapshots.
            float: The lambda value that achieved the best validation score.
        """
        best_score = -np.inf
        scores = []
        coeffs_list = []

        for lmbda in tqdm(self.lambdas, total=len(self.lambdas), disable=(verbose == 0)):
            if verbose == 2:
                print(f'Fitting lmbda: {lmbda}')
            self.fit(X_train, y_train, lmbda)
            score = self.validate(X_valid, y_valid, measure=measure)
            if verbose == 2:
                print(f'{measure.upper()} score: {score}')
            scores.append(score)
            coeffs_list.append(self.coefficients.copy())

            if score > best_score:
                best_score = score
                self.best_lambda = lmbda

        results = {
            'lambda': self.lambdas,
            measure: scores,
            'coefficients': coeffs_list
        }
        return results, self.best_lambda

    def plot(self, results, measure="f1"):
        """
        Plot performance measure against lambda values.

        Args:
            results (dict): Dictionary returned by the optimize_lambda method, containing lambda values and scores.
            measure (str): Performance measure to plot. Default is "f1".

        Returns:
            None

        Raises:
            ValueError: If results are empty or the specified measure is not found in the results.
        """
        if not results:
            print("No results to plot. First run optimize_lambda method.")
            return

        if measure not in results:
            print(f"Measure '{measure}' not found in results. Available measure: '{list(results.keys())[1]}'")
            print(f"Run optimize_lambda with measure='{measure}' to get these results.")
            return

        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(data={'lambda': results['lambda'], measure: results[measure]})
        df.plot(x='lambda', y=measure, logx=True)
        plt.grid(True)
        plt.xlabel('Lambda (log scale)')
        plt.ylabel(f'{measure.upper()} Score')
        plt.show()

    def plot_coefficients(self, results, aggregate=False):
        """
        Plot coefficient values for different lambda values.
        
        Args:
            results (dict): Dictionary returned by the optimize_lambda method, containing lambda values and coefficients.
            aggregate (bool): Whether to plot aggregated coefficient values. 
                            If True, plots the mean absolute value of all feature coefficients.
                            If False, plots individual feature coefficient paths.
        
        Returns:
            None

        Raises:
            ValueError: If results are empty or do not contain coefficient data.
        """
        if not results:
            print("No results to plot. First run optimize_lambda method.")
            return

        if 'coefficients' not in results:
            print("Coefficient data not found in results.")
            return

        plt.figure(figsize=(10, 6))
        lambdas = results['lambda']
        coeffs = results['coefficients']
        
        # Assume coeffs is a list of lists (or arrays) where the first element at index 0 is the intercept
        n_features = len(coeffs[0])
        
        if aggregate:
            # Aggregate feature coefficients only (excluding intercept at index 0)
            aggregated_coefs = [np.mean(np.abs(c[1:])) for c in coeffs]
            plt.plot(lambdas, aggregated_coefs)  # Legend removed by not adding labels
            ylabel = 'Mean Absolute Coefficient Value'
        else:
            # Plot individual feature coefficients, skipping intercept (index 0)
            for i in range(1, n_features):
                coef_values = [c[i] for c in coeffs]
                plt.plot(lambdas, coef_values)  # Legend removed by not adding labels
            ylabel = 'Coefficient Value'
        
        plt.xscale('log')
        plt.grid(True)
        plt.xlabel('Lambda (log scale)')
        plt.ylabel(ylabel)
        plt.show()


def generate_dataset(p=0.5, n=1000, d=10, g=0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthethic dataset

    Args: 
        p: prior probability for y=1
        n: number of instances
        d: number of features
        g: param for cov matrix

    Returns:
        X, y
    """
    y = bernoulli.rvs(p, size=n)
    
    # mean vectors
    m0 = np.zeros(d)
    m1 = np.array([1/(i+1) for i in range(d)])

    # cov matrix
    S = np.array([[g**abs(i - j) for j in range(d)] for i in range(d)])

    X = np.zeros((n, d))
    X[y==0] = multivariate_normal.rvs(mean = m0, cov=S, size=len(X[y==0]))
    X[y==1] = multivariate_normal.rvs(mean = m1, cov=S, size=len(X[y==1]))   

    return X, y