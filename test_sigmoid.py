import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# create data
x = np.linspace(-10, 10, 100)

# get sigmoid output
y = sigmoid(x)

# get derivative of sigmoid
d = d_sigmoid(x)


df = pd.DataFrame({"x": x, "sigmoid(x)": y, "d_sigmoid(x)": d})
# df.to_csv("sigmoid.csv", index=False)

plt.style.use("dark_background")

fig = plt.figure(figsize=(16, 9))

plt.plot(x, y, c="lightgreen", linewidth=3.0, label="$\sigma(x)$")
plt.plot(x, d, c="lightblue", linewidth=3.0, label="$\\frac{d}{dx} \sigma(x)$")

plt.legend(prop={'size': 20})
plt.show()