import pandas as pd

def main():
    # open dataset
    data = pd.read_csv("/home/fpelogia/Documentos/HMP/EstadoSP/SEADE/dados_covid_sp.csv", sep=';')

    # filter data from specific city
    df = data[data.nome_munic == 'Campinas']

    cols_to_keep = [
        'datahora', 
        'nome_munic', 
        'casos_novos', 
        'obitos_novos', 
        'casos', 
        'obitos',
    ]

    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    rename_dict = {
        'datahora' : 'date', 
        'nome_munic' : 'cidade', 
        'casos_novos' : 'new_confirmed',
        'obitos_novos' : 'new_deceased',
        'casos' : 'total_confirmed', 
        'obitos' : 'total_deceased',
    }
    
    df = df.rename(rename_dict, axis=1)

    # Save reduced dataset
    df.to_csv(f"Datasets/campinas.csv")

    print("\nSuccessful Extraction!")

if __name__ == "__main__":
    main()