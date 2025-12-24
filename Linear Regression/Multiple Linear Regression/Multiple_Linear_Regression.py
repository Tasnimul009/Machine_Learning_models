import numpy as np
class multiple_linear_regression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis = 1)
        betas = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        self.coef_ = betas[1:]
        self.intercept_ = betas[0]
    def predict(self, X):
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred


