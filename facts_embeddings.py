from dotenv import load_dotenv
import os
import logging
import sys
from typing import List
from datetime import date
import json
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from openai import OpenAI
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
import psycopg2
from psycopg2 import pool, extras
import numpy as np
import uuid

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

load_dotenv(verbose=True)

DB_HOST = os.getenv('DB_HOST')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PWD = os.getenv('DB_PWD')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')

openai_client = OpenAI()

def get_embedding_openai(vals):
    global openai_client
    if isinstance(vals, (np.ndarray, np.generic) ):
        vals = vals.tolist()
    response = openai_client.embeddings.create(
        input=vals,
        model='text-embedding-3-large'
    )
    return [emb_data.embedding for emb_data in response.data]


conn_string_db = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(DB_HOST, DB_USERNAME, DB_NAME, DB_PWD, 'require')
postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(1, 1, conn_string_db)

sql = "select id, fact, sentiment_score from theme.t_gdelt_gkg_news_facts where fact like '%%UAE%%'"

query = \
"""insert into theme.t_gdelt_gkg_news_facts_embeddings_large
(id, fact_embeddings_large, record_ingestion_timestamp)
values (%(id)s, %(fact_embeddings_large)s, %(record_ingestion_timestamp)s)
"""
if(postgreSQL_pool):
    engine = sqlalchemy.create_engine(f'postgresql+psycopg2://{DB_USERNAME}:{DB_PWD}@{DB_HOST}:5432/{DB_NAME}')
    conn_sa = engine.connect()
    conn = postgreSQL_pool.getconn()
    df = pd.read_sql(sql, conn_sa)
    df = df.loc[df['sentiment_score'] > 0]
    df = df.sample(n=700, random_state=42)
    df_chunks = np.array_split(df, 50)
    ldf = len(df_chunks)
    cnt = 0
    for df_chunk in df_chunks:
        df_chunk['record_ingestion_timestamp'] = pd.to_datetime('now')
        try:
            out_vals = get_embedding_openai(df_chunk['fact'].values)
        except Exception as ex:
            print(str(ex))
            continue
        df_chunk['fact_embeddings_large'] = out_vals
        print(f'Dataframe with {len(df_chunk)} rows')
        df_to_write = df_chunk[['id', 'fact_embeddings_large', 'record_ingestion_timestamp']]
        with conn.cursor() as cur:
            # insert data into theme layer
            data = df_to_write.to_dict(orient='records')
            try:
                psycopg2.extras.execute_batch(cur, query, data)
                conn.commit()
            except Exception as ex:
                print(str(ex))
        cnt += 1
        print(f'Processed {cnt} from {ldf}')
    postgreSQL_pool.putconn(conn)
    postgreSQL_pool.closeall()
