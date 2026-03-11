import numpy as np
class  multiple_linear_regression:
    def __init__(self):
        self.coef_ = None 
        self.intercept_ = None 
    def fit(self, X, y):
        X = np.insert(arr = X, 
                      obj = 0, 
                      values = 1, 
                      axis = 1)
        # [Xt*X]-1 * Xt*Y
        betas = np.linalg.inv(X.T @ X) @ X.T @ y 
        self.coef_ = np.array(betas[1:])
        self.intercept_ = betas[0]
    def predict(self, X):
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred 