import numpy as np
class BGD:
    def __init__(self, lr = 0.001, epochs = 100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = epochs
    def fit(self, X, y):
        self.coef_ = 1
        self.intercept_ = 0
        for i in range(self.epochs):
            m = self.coef_
            b = self.intercept_
            slope_m = -2*np.sum((y - m * X.ravel() - b) * X)
            slope_b = -2*np.sum(y - m * X.ravel() - b)
            m -= self.lr * slope_m
            b -= self.lr * slope_b
            self.coef_ = m
            self.intercept_ = b
    def predict(self, X):
        return self.coef_ * X + self.intercept_
