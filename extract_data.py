import pandas as pd

def main():
    # open dataset
    #data = pd.read_csv("../../Israel/cidades_israel.csv")
    data = pd.read_csv("../../Japan/japao_estados.csv")

    # ['Hokkaido' 'Aomori' 'Iwate' 'Miyagi' 'Akita' 'Yamagata' 'Fukushima'
    #  'Ibaraki' 'Tochigi' 'Gunma' 'Saitama' 'Chiba' 'Tokyo' 'Kanagawa'
    #  'Niigata' 'Toyama' 'Ishikawa' 'Fukui' 'Yamanashi' 'Nagano' 'Gifu'
    #  'Shizuoka' 'Aichi' 'Mie' 'Shiga' 'Kyoto' 'Osaka' 'Hyōgo' 'Nara'
    #  'Wakayama' 'Tottori' 'Shimane' 'Okayama' 'Hiroshima' 'Yamaguchi'
    #  'Tokushima' 'Kagawa' 'Ehime' 'Kōchi' 'Fukuoka' 'Saga' 'Nagasaki'
    #  'Kumamoto' 'Ōita' 'Miyazaki' 'Kagoshima' 'Okinawa']

    # Most populous prefectures in Japan
    #['Tokyo','Kanagawa','Osaka','Aichi','Saitama','Chiba','Hyōgo','Hokkaido','Fukuoka','Shizuoka']

    # filter data from specific city
    #df = data[data.cidade == 'Jerusalem']
    #df = data[data.estado == 'Tokyo']
    #df = data[data.estado == 'Kanagawa']
    #df = data[data.estado == 'Osaka']
    #df = data[data.estado == 'Aichi']
    #df = data[data.estado == 'Saitama']
    #df = data[data.estado == 'Chiba']
    #df = data[data.estado == 'Hyōgo']
    df = data[data.estado == 'Hokkaido']
    #df = data[data.estado == 'Fukuoka']
    df = data[data.estado == 'Shizuoka']
    

    # cols_to_keep = [
    #     'date', 
    #     'cidade', 
    #     'new_confirmed', 
    #     'new_deceased', 
    #     'new_recovered', 
    #     'total_confirmed', 
    #     'total_deceased', 
    #     'total_recovered'
    # ]
    
    
    cols_to_keep = [
        'date', 
        'estado', 
        'new_confirmed', 
        'new_deceased', 
        'new_recovered', 
        'cumulative_confirmed', 
        'cumulative_deceased', 
        'cumulative_recovered'
    ]
    

    # Remove unnecessary columns
    df.drop(df.columns.difference(cols_to_keep), 1, inplace=True)

    # Save reduced dataset
    df.to_csv("Datasets/hokkaido.csv")

    print("\nSuccessful Extraction!")



if __name__ == "__main__":
    main()