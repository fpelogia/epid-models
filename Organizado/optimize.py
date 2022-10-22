import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from epid_model import model, model_daily, share_variables 
# imports for debugging purposes
import readline
import code
import lmfit

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
    return x[2] - 1e-3
def constr4(x):
    # nu > 0.1
    return x[3] - 1e-3

initial_cond = lambda y_t : [] 

update_cond = lambda tp0A0 : [] 

def fit_data(acc_data_p, daily_data_p, city_name, x_nw, indicator='cases', n_weeks_pred = 2, scaling_factor = 1):
    global acc_data
    global daily_data
    global n_days
    global n_sig
    global sig_params
    global t

    acc_data = acc_data_p
    daily_data = daily_data_p

    t = np.linspace(0, len(acc_data)-1, len(acc_data))

    # Define constraints
    con1 = {'type':'ineq', 'fun':constr1}
    con2 = {'type':'ineq', 'fun':constr2}
    con3 = {'type':'ineq', 'fun':constr3}
    con4 = {'type':'ineq', 'fun':constr4}     
    
    cons = [con1, con2, con3, con4] 
    #cons = [con1, con2, con3] 

    #n_weeks_pred = 0
    n_sig = 1
    sig_params = []
    rel_rmse_list = []

    # set font-size
    plt.rcParams.update({'font.size': 12})

    fig, axs = plt.subplots(len(x_nw), 2, figsize=(14,12 + 3*(n_sig % 3)))
    fig.suptitle(f'{city_name}')

    for i in range(len(x_nw)):
        n_days = x_nw[i] - 7*n_weeks_pred

        share_variables(n_days, n_sig, sig_params)
        #print(f'========= Wave nr {i + 1} =========')
        #print('From 0 to ', n_days)

        #print('Step 1')
        # Step 1 - Optimize a symmetric sigmoid (nu = 1)
        # Initial values
        if(i == 0):
            y_t = acc_data[:n_days]            
            [A0, tp0, delta0, nu0] = initial_cond(y_t)
        else:
            [A0, tp0] = update_cond(A0, tp0)

        x0 = [A0, tp0, delta0, nu0]
        sol = minimize(loss_f_sym, x0, constraints=cons, args=('MSE'), method='SLSQP')
        
        #print(sol)

        # Optimal values
        [A, tp, delta, nu] = sol.x

        #print('Step 2')
        # Step 2 - Optimize an assymmetric sigmoid
        # using optimal values of step 1 as the starting point
        [A0, tp0, delta0, nu0] = sol.x

        x0 = [A0, tp0, delta0, nu0]
        
        if(n_sig == 1):
            sol = minimize(loss_f, x0, constraints=cons, args=('MSE'), method='SLSQP')
        else:
            sol = minimize(loss_f_sym, x0, constraints=cons, args=('MSE'), method='SLSQP')

        # Relative RMSE   (np.sqrt(MSE)/max(acc_data))
        rel_rmse = np.sqrt(sol.fun) / max(acc_data[:n_days])
        rel_rmse_list.append(f'{round(100*rel_rmse, 3)}%')
        #print('rRMSE: ', rel_rmse)

        # Optimal values
        [A, tp, delta, nu] = sol.x

        # prediction interval
        n_days = x_nw[i]

        share_variables(n_days, n_sig, sig_params)

        y_t = acc_data[:n_days]
        y_m0 = model(t[:n_days], A0, tp0, delta0, nu0)
        y_m = model(t[:n_days], A, tp, delta, nu)
        y_m_daily = model_daily(t[:n_days], A, tp, delta, nu)
        s = "" if (n_sig == 1) else "s"

        # Plotting Model vs Data

        #fig, axs = plt.subplots(1, 2, figsize=[15,5])
        #plt.xlim(x_nw[i] - 7*(n_weeks_pred + 1), x_nw[i])
        axs[i][0].scatter(t[:n_days], scaling_factor *acc_data[:n_days], label='Data', c='gray')
        if (n_weeks_pred > 0):
            axs[i][0].vlines(n_days - 7*n_weeks_pred, 0, scaling_factor * max(acc_data[:n_days]), colors='dimgray', linestyles='dashdot', zorder=1, label=f"Last {7*n_weeks_pred} days")
        axs[i][0].plot(scaling_factor * y_m, label='Model', c='r')
        axs[i][0].set_xlabel('t (days)')
        axs[i][0].set_ylabel(f'acc. number of {indicator}')
        
        axs[i][0].text(0, scaling_factor * 0.97 * max(y_t), f'rRMSE: {round(100*rel_rmse, 3)}%')

        if (n_weeks_pred > 0):
            X_detail = t[n_days - 7*n_weeks_pred: n_days]
            Y_detail = y_m[n_days - 7*n_weeks_pred: n_days]

            # Relative RMSE for the predictions
            y_t_pred = acc_data[n_days - 7*n_weeks_pred:n_days]
            y_m_pred = Y_detail
            mse_pred = (1/len(y_t_pred))*np.sum((y_t_pred - y_m_pred)**2)
            rel_rmse_pred = np.sqrt(mse_pred) / max(acc_data[:n_days])

            #print('rRMSE Predictions: ', rel_rmse_pred)

            axs[i][0].text(0, scaling_factor *0.90*max(y_t), f'rRMSE Predictions: {round(100*rel_rmse_pred, 3)}%')

            # detail prediction
            if(n_sig < 4):
                sub_axes = axs[i][0].inset_axes([.17, .45, .25, .35]) 
            else:
                sub_axes = axs[i][0].inset_axes([.6, .15, .25, .25]) 
            sub_axes.scatter(t[n_days - 7*n_weeks_pred:n_days], scaling_factor *acc_data[n_days - 7*n_weeks_pred:n_days], label='Data', c='gray')
            sub_axes.plot(X_detail, scaling_factor *Y_detail, c = 'r') 
            #sub_axes.set_xticks(X_detail[0::3])
            axs[i][0].indicate_inset_zoom(sub_axes, edgecolor="black")    
            #plt.savefig(f'output/Acc_{city_name}_{n_sig}_sig', facecolor='white', dpi=100)


        # Plotting Daily Data

        #axs[i][1].set_title(f'{city_name} - Model x Daily data')
        axs[i][1].plot(scaling_factor * np.array(daily_data[:n_days]), label="Data", c='gray', lw=0.8, linestyle='dashed')
        axs[i][1].plot(scaling_factor * np.array(y_m_daily), label='Model', c='r')
        if (n_weeks_pred > 0):
            axs[i][1].vlines(n_days - 7*n_weeks_pred, 0, scaling_factor * max(daily_data[:n_days]), colors='dimgray', linestyles='dashdot', zorder=1, label=f"Last {7*n_weeks_pred} days")
        axs[i][1].set_xlabel('t (days)')
        axs[i][1].set_ylabel(f'daily number of {indicator}')
        axs[i][1].legend(loc=2) # upper left    
        n_sig += 1
        sig_params.append([A, tp, delta, nu])
        #print(f'Parameters: {sig_params}\n==================================')    

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    #plt.savefig(f'output/Daily_{city_name}_2w_pred', facecolor='white', dpi=100)
    #plt.savefig(f'ESTADOSP/{city_name}', facecolor='white', dpi=200)
    plt.show(block=False)

    return sig_params, rel_rmse_list




#====== Interactive Debug ======
# variables = globals().copy()
# variables.update(locals())
# shell = code.InteractiveConsole(variables)
# shell.interact()
#===============================
