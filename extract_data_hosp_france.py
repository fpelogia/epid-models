import pandas as pd

def main():
    # open dataset
    data = pd.read_csv("../../França/hosp_franca_estados.csv")

    # filter data from specific city
    df = data[data.estado == 'Normandy']

    cols_to_keep = [
        'date', 
        'estado', 
        'current_hospitalized_patients',
        'current_intensive_care_patients'
    ]
    
    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/normandy_hosp.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()