from core.inventory import build_dataset, get_dataset_csv
import pandas as pd


if __name__ == "__main__":    
    df_path = get_dataset_csv()
    df = pd.read_csv(df_path)
    build_dataset(df)