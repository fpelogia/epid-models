import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
# imports for debugging purposes
import readline
import code

filter_data = lambda data : [] 

# Forward Euler Approximation (Euler's Method)
def forward_euler(t, step, y, dy0):
    dy = np.zeros(len(t))
    dy[0] = dy0
    for i in range(len(t) - 1):
        dy[i+1] = (y[i+1] - y[i])/step
    return dy

# New epidemiological wave detection
#   Receives (sec_der, abs_threshold)
#      sec_der: second derivative of acc. number of cases 
#      abs_threshold: threshold to consider around zero
#   Returns (x_t, y_t) -> coordinates of the transition points 
def new_wave_detection(sec_der, abs_threshold):
    x_t = []
    y_t = []
    for i in range(len(sec_der)-1):
        if((sec_der[i] < abs_threshold) and (sec_der[i+1] > abs_threshold)):
           x_t.append(i+1)
           y_t.append(sec_der[i+1])
    return x_t, y_t

# Get transition points
#  Receives (data): accumulated indicator
#  optional (visual): show transition points graph ?
#  optional (city_name): city name 
#  optional (threshold): threshold 
def get_transition_points(data, visual=False, city_name = "", threshold = 3e-5, indicator = 'cases'):    
    # Normalize by maximum value
    normalized_acc_n_cases = data / max(data)

    t = np.linspace(0, len(normalized_acc_n_cases), len(normalized_acc_n_cases))
    daily_n_cases = forward_euler(t, 1, normalized_acc_n_cases, 0)

    # Filter data to reduce noise effects
    unf_daily_n_cases = daily_n_cases
    daily_n_cases = filter_data(daily_n_cases)

    # Obtain second derivative of the number of cases w.r.t time
    # using Forward Euler
    t = np.linspace(0, len(daily_n_cases), len(daily_n_cases))
    sd0 = daily_n_cases[1] - daily_n_cases[0]
    sec_der = forward_euler(t, 1, daily_n_cases, sd0)

    # Detection of new waves
    abs_threshold = threshold
    x_t, y_t = new_wave_detection(sec_der, abs_threshold)

    if(visual):        
        # Graph with acc. data and its first two derivatives
        fig, axs = plt.subplots(3, 1, figsize=(11,13)) # 3 rows, 1 col
        plt.tight_layout(pad=1.5)
        #plt.suptitle(f"{city_name} threshold {abs_threshold} ", fontsize=16)
        plt.suptitle(f"Pontos de transição - {city_name} - threshold {abs_threshold} ", fontsize=16)
        axs[0].plot(normalized_acc_n_cases) # para alinhar as retas de nova onda
        axs[0].vlines(x_t, 1, 3e-4, colors='dimgray', linestyles='dashdot', zorder=1, label="new wave transition")
        #axs[0].set_title(f'Normalized accumulated number of {indicator}')
        axs[0].set_title(f'Número acumulado de casos normalizado')
        #axs[0].set_ylabel(f"${indicator}$")
        axs[0].set_ylabel(f"$casos$")

        axs[1].plot(unf_daily_n_cases, c='darkgray')
        axs[1].plot(daily_n_cases)
        axs[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2))
        axs[1].vlines(x_t, min(unf_daily_n_cases), max(unf_daily_n_cases), colors='dimgray', linestyles='dashdot', zorder=1, label="new wave transition")
        axs[1].set_title('Primeira derivada')
        #axs[1].set_ylabel(f"${indicator}$ / $day$")
        axs[1].set_ylabel(f"$casos / dia$")

        axs[2].ticklabel_format(axis='y',style='sci',scilimits=(-4,-4))
        axs[2].set_title("Segunda derivada - Detecção de novas ondas")
        axs[2].set_xlabel("t (dias)")
        #axs[2].set_ylabel(f"${indicator}$ / $day^2$")
        axs[2].set_ylabel(f"$casos$ / $dia^2$")
        axs[2].plot(sec_der, zorder=1) # obs: check if this scaling is correct
        axs[2].hlines([-1*abs_threshold, abs_threshold], 0, len(sec_der), colors='silver', linestyles='dashed', zorder=1, label=f"threshold = $\pm${abs_threshold}")
        axs[2].vlines(x_t, -3e-4, 3e-4, colors='dimgray', linestyles='dashdot', zorder=1, label="transição para nova onda")
        axs[2].scatter(x_t, y_t, s=15, c='r', zorder=2, label="mudança de sinal na segunda derivada")
        plt.legend()
        plt.tight_layout()
        #plt.savefig(f'{city_name}_nw', facecolor='white', dpi=200)
        plt.savefig(f'Figuras/TG_T1_NW_{city_name}', facecolor='white', dpi=200)
        plt.show(block=False)
    return x_t
