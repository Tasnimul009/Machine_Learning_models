import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=20
)

fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolor='k')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gd(X, y, lr=0.1, epochs=1000):
    m = []
    b = []
    X = np.insert(X, 0, 1, axis = 1)
    weights = np.ones(X.shape[1])
    for _ in range(epochs):
        Z = np.dot(X, weights)
        y_hat = sigmoid(Z)
        error = y - y_hat
        weights += lr * np.dot(error, X) / X.shape[0]
        m.append(-weights[1] / weights[2])
        b.append(-weights[0] / weights[2])
    return m, b

m, b = gd(X, y, lr=0.1, epochs=1000)

Xi = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yi = m[0] * Xi + b[0]
line, = ax.plot(Xi, yi, color='red', linewidth=2)

def update(frame):
    yi = m[frame] * Xi + b[frame]
    line.set_ydata(yi)
    ax.set_xlabel(f"Epoch {frame+1}")
    return line,


anim = FuncAnimation(fig, update, frames=len(m), interval=10, repeat=False)


plt.title("Gradient Descent Decision Boundary Animation")
plt.ylabel("Feature 2")
plt.show()
