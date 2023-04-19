import numpy as np
from lmfit import minimize, Parameters

def richards_sigmoid(t, A, tp, delta, nu):
    return A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))

def sum_richards_sigmoids(t, num_sigmoids, *sig_params):
    y = np.zeros_like(t)
    for i in range(num_sigmoids):
        A, tp, delta, nu = sig_params[i*4:(i+1)*4]
        y += richards_sigmoid(t, A, tp, delta, nu)
    return y

def mse(params, t, y):
    num_sigmoids = params['num_sigmoids']
    sig_params = [params[f'sig_{i}'] for i in range(num_sigmoids * 4)]
    y_pred = sum_richards_sigmoids(t, num_sigmoids, *sig_params)
    return np.mean((y - y_pred)**2)

def fit_richards_sigmoids(t, y, num_sigmoids, initial_guesses=None):
    params = Parameters()
    params.add('num_sigmoids', value=num_sigmoids, vary=False)
    if initial_guesses is None:
        initial_guesses = [100, 1, 1, 1] * num_sigmoids
    for i in range(num_sigmoids):
        params.add(f'sig_{i}_A', value=initial_guesses[i*4])
        params.add(f'sig_{i}_tp', value=initial_guesses[i*4+1])
        params.add(f'sig_{i}_delta', value=initial_guesses[i*4+2])
        params.add(f'sig_{i}_nu', value=initial_guesses[i*4+3])
    result = minimize(mse, params, args=(t, y))
    return result
