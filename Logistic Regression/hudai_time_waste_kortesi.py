import numpy as np
import plotly.graph_objects as go

# Define range
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create 2D grid
xg, yg = np.meshgrid(x, y)

# Compute z
z = xg**3 + yg**3

# Plot
fig = go.Figure(go.Surface(x=xg, y=yg, z=z))
fig.add_trace(go.Surface(x=xg, y=yg, z=z**2))
fig.show()
