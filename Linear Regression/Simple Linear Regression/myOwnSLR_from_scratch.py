import numpy as np
class simpleLinearRegression:
    def __init__(self):
        self.coef_ = None  
        self.intercept_ = None
    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        num = np.sum((X - X.mean()) * (y - y.mean()))
        den = np.sum((X - X.mean()) ** 2)
        self.coef_ = num/den
        self.intercept_ = y.mean() - self.coef_ * X.mean()
    def predict(self, X):
        X = X.flatten()
        return self.coef_*X + self.intercept_