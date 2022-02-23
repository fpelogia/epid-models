import numpy as np
import matplotlib.pyplot as plt

'''
Richards Model (Assymetric Sigmoid)
t: time array
A, tp, delta, nu: model parameters
'''
def model(t, A, tp, delta, nu ):
    return A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))

# Vectorized version of the model
model_v = np.vectorize(model)

def main():
    # Define time array
    t = np.arange(-50, 50, 1)
    # Initialize model parameters
    A = 3
    tp = 7
    delta = 2
    nu = 7

    #Plot model's output
    plt.plot(model_v(t, A, tp, delta, nu))
    plt.show()

if __name__ == '__main__':
    main()