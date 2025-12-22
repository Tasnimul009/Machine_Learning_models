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
    class_sep=10
)

fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolor='k')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perceptron(X, y, lr=0.01, epochs=1000):
    m_list = []
    b_list = []

    # add bias column
    X_aug = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X_aug.shape[1])

    for _ in range(epochs):
        i = np.random.randint(0, X_aug.shape[0])
        z = np.dot(weights, X_aug[i])
        y_hat = sigmoid(z)
        error = y[i] - y_hat
        weights += lr * error * X_aug[i]
        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]
        m_list.append(slope)
        b_list.append(intercept)

    return m_list, b_list

m, b = perceptron(X, y, lr=0.1, epochs=1000)

Xi = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yi = m[0] * Xi + b[0]
line, = ax.plot(Xi, yi, color='red', linewidth=2)

def update(frame):
    yi = m[frame] * Xi + b[frame]
    line.set_ydata(yi)
    ax.set_xlabel(f"Epoch {frame+1}")
    return line,


anim = FuncAnimation(fig, update, frames=len(m), interval=10, repeat=False)


plt.title("Perceptron Decision Boundary Animation")
plt.ylabel("Feature 2")
plt.show()
