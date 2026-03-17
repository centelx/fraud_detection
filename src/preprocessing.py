import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from src.data_loader import load_data_from_db
from src.logger import get_logger

logger = get_logger(__name__)

def get_processed_data(db_url='postgresql://postgres:root@localhost:5432/fraud_db'):
    """loading, scaling and splitting data for training and testing sets."""

    df = load_data_from_db(db_url)

    logger.info(" scaling ")
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    logger.info(" splitting ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"X_train: {len(X_train)} rows, X_test: {len(X_test)} rows.")

    return X_train, X_test, y_train, y_test