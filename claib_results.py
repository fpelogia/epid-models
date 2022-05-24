import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# sigmoid.py (https://github.com/fpelogia/epid-models/blob/master/sigmoid.py)
from sigmoid import f_t, deriv_f_t 
# new_wave.py (https://github.com/fpelogia/epid-models/blob/master/new_wave.py)
from new_wave import new_wave_detection, filter_data, forward_euler, moving_average 
import readline
import code

# Global variables
n_sig = 1
n_days = 0
sig_params = []
acc_data = []
t = []

def model(t, A, tp, delta, nu):
    res = np.zeros(n_days)
    for i in range(n_sig - 1):
        [A_i, tp_i, delta_i, nu_i] = sig_params[i]
        res += f_t(t[:n_days], A_i, tp_i, delta_i, nu_i)

    res += f_t(t[:n_days], A, tp, delta, nu)
    return res

def model_daily(t, A, tp, delta, nu):
    res = np.zeros(n_days)
    for i in range(n_sig - 1):
        [A_i, tp_i, delta_i, nu_i] = sig_params[i]
        res += deriv_f_t(t[:n_days], A_i, tp_i, delta_i, nu_i)

    res += deriv_f_t(t[:n_days], A, tp, delta, nu)
    return res

# Integral Time Square Error (ITSE)
def ITSE(x):
    # model parameters
    A = x[0]
    tp = x[1]
    delta = x[2]
    nu = x[3]

    y_t = acc_data[:n_days]
    y_m = model(t[:n_days], A, tp, delta, nu)
    return np.sum(t[:n_days]*(y_t - y_m)**2)

# Mean Squared Error (MSE)
def MSE(x):
    # model parameters
    A = x[0]
    tp = x[1]
    delta = x[2]
    nu = x[3]

    y_t = acc_data[:n_days]
    y_m = model(t[:n_days], A, tp, delta, nu)
    return (1/len(y_t))*np.sum((y_t - y_m)**2)

def loss_f(x, lf):
    if(lf == 'MSE'):
        return MSE(x)
    elif(lf == 'ITSE'):
        return ITSE(x)
    else:
        return MSE(x)


def loss_f_sym(x, lf):
    # nu = 1 (symmetric sigmoid)
    if(lf == 'MSE'):
        return MSE([x[0], x[1], x[2], 1]) 
    elif(lf == 'ITSE'):
        return ITSE([x[0], x[1], x[2], 1]) 
    else:
        return MSE([x[0], x[1], x[2], 1]) 
    

# Inequality contraints need to return f(x), where f(x) >= 0
def constr1(x):
    # A >= 0
    return x[0]
def constr2(x):
    # tp >= 0
    return x[1]
def constr3(x):
    # delta >= 0.1
    return x[2] - 1e-1
def constr4(x):
    # nu > 0.1
    return x[3] - 1e-1

con1 = {'type':'ineq', 'fun':constr1}
con2 = {'type':'ineq', 'fun':constr2}
con3 = {'type':'ineq', 'fun':constr3}
con4 = {'type':'ineq', 'fun':constr4}     
cons = [con1, con2, con3, con4] 

def main():
    global t
    global acc_data
    global daily_data
    global sig_params
    global n_sig
    global n_days
    # Increase font-size
    plt.rcParams.update({'font.size': 12})

    # Import data
    data = pd.read_csv("Datasets/mendoza_obitos.csv") 
    city_name = 'Mendoza' 

    acc_data = data.cumulative_deceased
    normalized_acc_data = acc_data / max(acc_data)
    t = np.linspace(0, len(acc_data)-1, len(acc_data))
    
    normalized_acc_data = normalized_acc_data.tolist()
    daily_data = data.new_deceased.tolist()


    # Transition Points

    if city_name == 'Rosario':
        x_nw = [295, 360, 515] # Manual Rosario
    elif city_name == 'Buenos Aires':
        x_nw = [300, 400, 520] # Manual Buenos Aires
    elif city_name == 'Mendoza':
        x_nw = [315, 400, 510] # Manual Mendoza
    elif city_name == 'Córdoba':
        x_nw = [300, 400, 520] # Manual Córdoba
    else:
        x_nw = [300, 400, 520] 

    # Predictions for the first three waves
    n_weeks_pred = 2
    n_sig = 1
    sig_params = []

    fig, axs = plt.subplots(3, 1, figsize=(10,16))
    #fig.suptitle(f'{city_name} - Model x Daily number of deaths')
    fig.suptitle(f'{city_name}')
    for i in range(len(x_nw)):
        x_nw[i]
        n_days = x_nw[i]- 7*n_weeks_pred
        print(f'========= Wave nr {i + 1} =========')
        print('From 0 to ', n_days)
        print('Step 1')
        # Step 1 - Optimize a symmetric sigmoid (nu = 1)
        # Initial values
        if(i == 0):
            y_t = acc_data[:n_days]
            A0 = 2*max(y_t)
            tp0 = (2/3)*len(y_t)
            delta0 = (1/4)*len(y_t)
            nu0 = 1
        else:
            tp0 += 100
            A0 *= 0.05

        x0 = [A0, tp0, delta0, nu0]
        sol = minimize(loss_f_sym, x0, constraints=cons, args=('MSE'), method='SLSQP')
        print(sol)

        # Optimal values
        [A, tp, delta, nu] = sol.x

        print('Step 2')
        # Step 2 - Optimize an assymmetric sigmoid
        # using optimal values of step 1 as the starting point
        [A0, tp0, delta0, nu0] = sol.x

        x0 = [A0, tp0, delta0, nu0]
        sol = minimize(loss_f, x0, constraints=cons, args=('MSE'), method='SLSQP')
        print(sol)

        # Optimal values
        [A, tp, delta, nu] = sol.x

        # due to filtering delay
        n_days = x_nw[i]

        y_t = acc_data[:n_days]
        y_m0 = model(t[:n_days], A0, tp0, delta0, nu0)
        y_m = model(t[:n_days], A, tp, delta, nu)
        y_m_daily = model_daily(t[:n_days], A, tp, delta, nu)
        s = "" if (n_sig == 1) else "s"

        if (city_name == 'Buenos Aires'):
            shift_val = 21 # Buenos Aires
        if (city_name == 'Mendoza'):
            shift_val = 7 # Mendoza
        if (city_name == 'Rosario'):
            shift_val = 11 # Rosario
        if (city_name == 'Córdoba'):
            shift_val = 8 # Córdoba

        axs[i].plot(daily_data[shift_val:n_days], label="Data", c='gray', lw=0.8, linestyle='dashed')
        axs[i].plot(y_m_daily[shift_val:], label='Model', c='r')
        axs[i].vlines(n_days - 7*n_weeks_pred - shift_val, 0, max(daily_data[:n_days]), colors='dimgray', linestyles='dashdot', zorder=1, label=f"Last {7*n_weeks_pred} days")
        axs[i].set_xlabel('t (days)')
        axs[i].set_ylabel('Daily number of deaths')
        axs[i].legend(loc=2) # upper left

        n_sig += 1
        sig_params.append([A, tp, delta, nu])
        print(f'===================================')    

    plt.tight_layout()
    plt.savefig(f'CLAIB/{city_name}', facecolor='white', dpi=100)
    plt.show()

if __name__ == "__main__":
    main()

#====== Interactive Debug ======
# variables = globals().copy()
# variables.update(locals())
# shell = code.InteractiveConsole(variables)
# shell.interact()
#===============================