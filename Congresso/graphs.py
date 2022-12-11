import numpy as np
import matplotlib.pyplot as plt
from new_wave import forward_euler

def exp_t(C0, alpha, t):
    return C0*np.exp(alpha*t)
exp_t = np.vectorize(exp_t)

def der_exp_t(alpha, t):
    return alpha*t
der_exp_t = np.vectorize(der_exp_t)

def logist_t(alpha, tp, A, t):
    return A/(1 + np.exp(-alpha*(t - tp)))
exp_t = np.vectorize(exp_t)

def richards_t(t, A, tp, delta, nu ):
    return A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))

richards_t = np.vectorize(richards_t)

def main():
    # Definições iniciais
    t = np.linspace(0, 12, 50)
    fig, axs = plt.subplots(2, 1, figsize=(5,8))
    fig.suptitle('Richards Model')
    #C_t = logist_t(0.8, 6, 10, t) #Logístico
    C_t = richards_t(t, 10, 6, 0.6, 3)
    #C_t = exp_t(1, 0.5, t)
    axs[0].plot(C_t[2:])
    axs[0].set_xlabel('t (days)')
    axs[0].set_ylabel('D(t)')

    der = forward_euler(t, 1, C_t, 0)
    axs[1].plot(der[2:])
    axs[1].set_xlabel('t (days)')
    axs[1].set_ylabel(r'$\frac{dD}{dt}(t)$')
    plt.tight_layout()
    plt.show() 
    #plt.savefig('richards-model.png')

if __name__ == "__main__":
    main()
