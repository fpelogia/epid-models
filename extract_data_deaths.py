import pandas as pd

def main():
    # open dataset

    # Rosario:
    #data = pd.read_csv("../../Argentina/obitos_argentina_cidades.csv")
    # City of Buenos Aires, Mendoza, Córdoba
    data = pd.read_csv("../../Argentina/obitos_argentina_estados.csv")
    # filter data from specific city
    df = data[data.estado == 'Córdoba']

    cols_to_keep = [
        'date', 
        'estado',  # cidade (Rosario)
        'new_deceased',
        'cumulative_deceased'
    ]
    
    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/cordoba_obitos.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()