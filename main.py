from src.data_ingestion import load_and_sample_data
from src.trainer import train_model

if __name__ == "__main__":
    df = load_and_sample_data("data/tweets.csv", samples_per_class=20000)
    trainer = train_model(df)
