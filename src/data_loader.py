import pandas as pd
from sqlalchemy import create_engine
import time
from src.logger import get_logger

logger = get_logger(__name__)

def upload_csv_to_db(csv_path, db_url):
    df = pd.read_csv(csv_path)

    engine = create_engine(db_url)
    df.to_sql('transactions', engine, if_exists='replace', index=False, chunksize=10000)
    logger.info(f"data loaded from csv to db")

def load_data_from_db(db_url):
    logger.info("loading data from db")
    engine = create_engine(db_url)
    query = 'SELECT * FROM transactions;'
    df = pd.read_sql(query, engine)
    return df