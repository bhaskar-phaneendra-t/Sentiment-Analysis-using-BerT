import pandas as pd
from src.logger import get_logger

logger = get_logger()

def load_and_sample_data(csv_path, samples_per_class=20000):
    logger.info("Loading dataset")
    df = pd.read_csv(csv_path, encoding="latin1", header=None)

    df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    logger.info("Sampling balanced dataset")
    df = (
        df.groupby("sentiment", group_keys=False)
        .apply(lambda x: x.sample(samples_per_class, random_state=42))
    )

    return df
