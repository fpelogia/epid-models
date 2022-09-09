import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimize import fit_data
from shared_vars import *

def main():
    # Import data
    data = pd.read_csv("../Datasets/rosario_obitos.csv") 
    city_name = 'Rosario' 

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
    
    fit_data(acc_data, daily_data, city_name, x_nw)
    

if __name__ == "__main__":
    main()
