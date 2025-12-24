class simpleLinearRegression:
    def  __init__(self):
        self.m = None
        self.b = None
    def fit(self, X, y):
        num = 0
        den = 0
        for i in range(X.shape[0]):
            num += (X[i] - X.mean()) * (y[i] - y.mean())
            den += (X[i] - X.mean()) ** 2
        self.m = num/den
        self.b = y.mean() - self.m * X.mean()
    def predict(self, X):
        return self.m * X + self.b