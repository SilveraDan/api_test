# crypto_forecast_ml/data_loader.py

import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from google.oauth2 import service_account

# ⚙️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_crypto_data(symbol: str = "BTCUSDT", days: int = 7, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info("🟡 load_crypto_data() CALLED")

    gcp_credentials_json = os.environ["GCP_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials_json))
    bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    # Fenêtre temporelle
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Launching BigQuery query for {symbol}...")
    df = bq_client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")

    return df


def load_crypto_data_custom_range(symbol: str, start_date: str, end_date: str, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info(f"🟡 load_crypto_data_custom_range() CALLED with symbol={symbol}, start={start_date}, end={end_date}")

    # 🔍 Recherche les credentials
    gcp_credentials_json = os.environ["GCP_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials_json))
    bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
          AND timestamp_utc <= TIMESTAMP('{end_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Running query for range: {start_date} to {end_date}")
    logger.info(f"📥 Running query : {query}")
    df = bq_client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")
    return df
