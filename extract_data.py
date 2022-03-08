import pandas as pd

def main():
    # open dataset
    data = pd.read_csv("../../Israel/cidades_israel.csv")

    # filter data from specific city
    df = data[data.cidade == 'Jerusalem']

    cols_to_keep = [
        'date', 
        'cidade', 
        'new_confirmed', 
        'new_deceased', 
        'new_recovered', 
        'total_confirmed', 
        'total_deceased', 
        'total_recovered'
    ]
    
    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/jerusalem.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()