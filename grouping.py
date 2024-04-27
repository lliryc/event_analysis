import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import json
import time
import sqlalchemy
from sqlalchemy.sql import text
import umap.umap_ as umap
from dotenv import load_dotenv
from random import sample
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime, timezone
import traceback
import pandas as pd
import numpy as np
import os
import logging
import sys

load_dotenv()

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

DB_HOST = os.getenv('DB_HOST')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PWD = os.getenv('DB_PWD')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')

select_pos_sql = \
"""select f1.id, f1.news_url, f1.fact, f2.fact_embeddings_large as fact_embeddings, f1.sentiment_score from theme.t_gdelt_gkg_news_facts f1 
inner join theme.t_gdelt_gkg_news_facts_embeddings_large f2 on f1.id = f2.id 
where f1.sentiment_score > 0"""

select_neg_sql = \
"""select f1.id, f1.news_url, f1.fact, f2.fact_embeddings_large as fact_embeddings, f1.sentiment_score from theme.t_gdelt_gkg_news_facts f1 
inner join theme.t_gdelt_gkg_news_facts_embeddings_large f2 on f1.id = f2.id 
where f1.sentiment_score < 0"""


def sample_facts(items):
    litems = list(items)
    if len(litems) <= 10:
        return litems
    else:
        return sample(litems, 10)

def str2np(x):
    return np.fromstring(
        x.replace('\n', '')
            .replace('[', '')
            .replace(']', '')
            .replace('  ', ' '), sep=',')

openai_client = ChatOpenAI()

fact_summary_prompt_template = """
Please generate a headline in English in no more than 10 words that corresponds to facts below. If some facts are out of main context, ignore them:
{fact_list}
"""

def headline_gen_openai(facts, temperature=0.3, model_name="gpt-4"):
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    prompt = PromptTemplate(
        template=fact_summary_prompt_template,
        input_variables=["fact_list"]
    )
    b_facts = [f'{str(i)}. {t}'for i, t in enumerate(facts)]
    fact_list = '\n'.join(b_facts)
    chain = prompt | llm
    res = chain.invoke({"fact_list": fact_list})
    return res.content.strip('"')

def get_gkg_news_cluster_items(select_sql, eps=0.2, min_samples=5, top_size=10):
    url = f'postgresql+psycopg2://{DB_USERNAME}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    engine = sqlalchemy.create_engine(url)

    with engine.connect() as conn:
        query = conn.execute(text(select_sql))

    df = pd.DataFrame(query.fetchall())
    df = df.drop_duplicates(subset=['fact'])
    len_df = len(df)
    if len_df <= max(10, top_size):
        raise Exception(f'Not enough facts for grouping ({len_df} articles)')
    time0 = time.time()
    print('compressing dimensions')
    n_components = min(300, int(0.7 * len_df))
    # if n_components < 300:
    #    eps = 0.5 # by default
    umap_model = umap.UMAP(n_neighbors=8, n_components=n_components, min_dist=0.0, metric='cosine', random_state=42)
    df['fact_embeddings'] = df['fact_embeddings'].apply(lambda x: str2np(x))
    emb_array = df['fact_embeddings'].values
    emb_array = np.stack(emb_array)
    emb_array = umap_model.fit_transform(emb_array)
    print('compressing dimensions time:  count {} cost time: {}'.format(len(emb_array), time.time() - time0))
    # apply DBSCAN
    time1 = time.time()
    print('eps value: {}'.format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    clusters = dbscan.fit_predict(emb_array)
    print('dbscan  processed {} records, cost time: {}'.format(emb_array.shape, time.time() - time1))

    df['clusters'] = clusters
    df = df.loc[df['clusters'] > -1]
    df['index_val'] = df.index
    df_aggs = df.groupby(['clusters']).agg(
        id=('id', np.max),
        num=('id', 'count'),
        indices=('index_val', lambda x: sample_facts(x))
    )
    df_aggs['facts'] = df_aggs['indices'].apply(lambda x: df.loc[x]['fact'].values.tolist())
    res_df = pd.merge(df_aggs, df, 'inner', on=['id'])
    res_df = res_df.sort_values('num', ascending=False)
    res_df['rank'] = range(1, 1 + len(res_df))
    return res_df

def headline_extraction_general(select_sql):
    try:
        sql = select_sql
        eps = 0.20
        res_df = get_gkg_news_cluster_items(select_sql=sql, eps=eps, top_size=10)
        res_df['headline'] = res_df['facts'].apply(headline_gen_openai)
        res_df = res_df.loc[:, ['headline', 'num', 'rank']]
    except Exception as ex:
        traceback.print_exc()
        logging.error(ex)
        return None
    return res_df


if __name__ == '__main__':
    res = headline_extraction_general(select_pos_sql)
    print(res)