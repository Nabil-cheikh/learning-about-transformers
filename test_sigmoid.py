import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
# plt.show()

# create variable used for prediction
hours_spent_learning = np.linspace(1, 100, 100)

# probability of passing
weights = hours_spent_learning / len(hours_spent_learning)
# print(weights)

# create variable we want to predict from hours_spent_learning
pass_test = np.random.binomial(1, weights)
# print(pass_test)

x = hours_spent_learning.reshape(-1, 1)
y = pass_test

# fit the model
model = LogisticRegression()
model.fit(x, y)

# use the model coefficients to draw the plot
pred = sigmoid(x * model.coef_[0] + model.intercept_[0])


fig = plt.figure(figsize=(16, 9))
plt.plot(x, pred, c="lightblue", linewidth=3.0)
plt.scatter(
    x[(y == 1).ravel()],
    y[(y == 1).ravel()],
    marker=".",
    c="lightgreen",
    linewidth=1.0,
    label="passed",
)
plt.scatter(
    x[(y == 0).ravel()],
    y[(y == 0).ravel()],
    marker=".",
    c="red",
    linewidth=1.0,
    label="failed",
)
plt.axhline(y=0.5, color="orange", linestyle="--", label="boundary")
plt.xlabel("Hours spent learning")
plt.ylabel('p("passing the test")')
plt.legend(frameon=False, loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5), prop={'size': 20})
plt.show()
