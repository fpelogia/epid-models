import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def f_t(t, A, tp, delta, nu):
    return A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))

def richards_sum(t, *params):
    n = len(params) // 4
    result = np.zeros_like(t)
    for i in range(n):
        A, tp, delta, nu = params[i*4:i*4+4]
        result += A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))
    return result

# Define the objective function to minimize
def objective(params, t, y):
    return np.sum((richards_sum(t, *params) - y)**2)

# Generate some example data
t = np.linspace(0, 10, 100)
y = 2 / ((1 + np.exp(-1*(t - 5)))**(1/2)) + 1 / ((1 + np.exp(-1*(t - 3)))**(1/3)) + 0.5 / ((1 + np.exp(-1*(t - 8)))**(1/4)) + np.random.normal(0, 0.1, len(t))

# Apply StandardScaler to input data
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

# Initial guess for parameters
params0 = [2, 5, 1, 2, 1, 3, 1, 3, 0.5, 8, 1, 4]

# Minimize the objective function
res = minimize(objective, params0, args=(t, y_scaled))

# Print the optimized parameters
print(res.x)

# Generate plot comparing model output with input data
fig, ax = plt.subplots()
ax.plot(t, scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel(), 'o', label='Input data')
ax.plot(t, scaler.inverse_transform(richards_sum(t, *res.x).reshape(-1, 1)).ravel(), label='Model output')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()
