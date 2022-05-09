import pandas as pd

def main():
    # open dataset
    data = pd.read_csv("../../Argentina/obitos_argentina_cidades.csv")

    # filter data from specific city
    df = data[data.cidade == 'Rosario']

    cols_to_keep = [
        'date', 
        'cidade', 
        'new_deceased',
        'cumulative_deceased'
    ]
    
    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/rosario_obitos.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()