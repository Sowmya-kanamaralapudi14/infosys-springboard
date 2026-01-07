from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

X = np.array([
    [200, 300, 50, 250],
    [180, 400, 30, 260],
    [220, 350, 40, 270]
])

y = np.array([260, 280, 300])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "pricing_model.pkl")

print("DONE: pricing_model.pkl file created")