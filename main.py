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
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
from fastparquet import write, ParquetFile
from pandarallel import pandarallel
import psycopg2
from psycopg2 import pool, extras

pandarallel.initialize(progress_bar=True) # nb_workers

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

openai_client = ChatOpenAI()

def get_cop28_news():
    select_sql = """
    select t1.id, t1.record_id, t1.source_timestamp, t1.collect_timestamp, t1.news_source_type, t1.news_url, 
t1.ref_all_names, t1.ref_locations_detail, t1.ref_persons_detail, t1.ref_organizations_detail, 
t1.sharing_image, t1.news_theme, t1.news_title, 
t2.news_content, 
t1.ref_geo_loc, t1.ref_locations_fullname, t1.ref_persons_fullname, t1.ref_organizations_fullname, 
t1.news_sentiment_gdelt, t1.record_ingestion_timestamp from theme.t_gdelt_gkg as t1
inner join theme.t_gdelt_gkg_news_content as t2 on t1.id = t2.id  
where t1.news_title like '%%COP28%%';   
    """
    url = f'postgresql+psycopg2://{DB_USERNAME}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    engine = sqlalchemy.create_engine(url)

    with engine.connect() as conn:
        query = conn.execute(text(select_sql))

    df = pd.DataFrame(query.fetchall())
    df = df.drop_duplicates(subset=['news_title'])
    write('cop28_dataset/cop28_records.parq', df)
    return

def extract_facts():
    pf = ParquetFile('cop28_dataset/cop28_records.parq')
    df = pf.to_pandas()
    df['neg_score'] = df['news_sentiment_gdelt'].apply(lambda x: x['is_news_negative'])
    #df['pos_score'] = df['news_sentiment_gdelt'].apply(lambda x: x['is_news_positive'])
    df_neg = df.loc[df['neg_score'] > 3]
    rec_df = extract_neg_facts(df_neg)
    return rec_df

topic_summary_prompt_template = \
"""
Please provide only very negative sentiment facts about UAE (United Arab Emirates) from following article and present them as json array of items with field "fact" and "sentiment_score" with float value in interval [-1, 0] where -1 indicates the highest negative score. Otherwise, return empty json array:

{article}
"""

def neg_facts_gen_openai(row, temperature=0.3, model_name="gpt-4"):
    facts_recs = []
    try:
        llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=topic_summary_prompt_template,
            input_variables=["article"]
        )
        chain = prompt | llm | parser
        res = chain.invoke({"article": row["news_content"]})
        id = row["id"]

        record_ingestion_timestamp = row["record_ingestion_timestamp"]
        for fact_item in res:
            facts_recs.append(
                {"id": uuid.uuid4().hex, "news_id": id, "news_url": row["news_url"], "fact": fact_item["fact"],
                 "sentiment_score": fact_item["sentiment_score"],
                 "news_record_ingestion_timestamp": record_ingestion_timestamp,
                 })
    except Exception as ex:
        print(str(ex))
    return facts_recs

def extract_neg_facts(df):
    facts_recs_res = df.parallel_apply(neg_facts_gen_openai, axis=1)
    acc = []
    for facts_recs in facts_recs_res:
        for fact_rec in facts_recs:
            acc.append(fact_rec)

    res_df = pd.DataFrame.from_records(acc)
    return res_df

def save_facts(df):
    db_host = 'c-presight-mediasense-weur-cosmos-psql.dlrquzg4dybbgb.az.presight.ai'
    db_name = 'mediasense'
    db_user = 'citus'
    db_pw = 'Presight1234'
    conn_string_db = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(db_host, db_user, db_name,
                                                                                     db_pw, 'require')
    query = \
"""
INSERT INTO theme.t_gdelt_gkg_news_facts
(id, news_id, news_url, fact, sentiment_score, news_record_ingestion_timestamp)
VALUES (%(id)s, %(news_id)s, %(news_url)s, %(fact)s, %(sentiment_score)s, %(news_record_ingestion_timestamp)s)
"""
    postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(1, 1, conn_string_db)

    if (postgreSQL_pool):
        conn = postgreSQL_pool.getconn()
        with conn.cursor() as cur:
            # insert data into theme layer
            data = df.to_dict(orient='records')
            try:
                psycopg2.extras.execute_batch(cur, query, data)
                conn.commit()
            except Exception as ex:
                print(str(ex))
        postgreSQL_pool.putconn(conn)
        postgreSQL_pool.closeall()
    else:
        print("Can't save table, connection problems")

if __name__ == '__main__':
    df = extract_facts()
    save_facts(df)
