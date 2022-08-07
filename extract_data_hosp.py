import pandas as pd

def main():
    # open dataset
    data = pd.read_csv("../../Argentina/hosp_argentina_estados.csv")

    # filter data from specific city
    df = data[data.estado == 'City of Buenos Aires']

    cols_to_keep = [
        'date', 
        'estado', 
        'new_hospitalized_patients',
        'cumulative_hospitalized_patients'
    ]
    
    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/buenos_aires_hosp.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()