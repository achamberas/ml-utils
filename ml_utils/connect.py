'''
Set of functions to connect to google cloud, execute queries on google cloud and get simulation data / jobs.
'''

import os
import pandas as pd

import google.auth
from google.oauth2 import service_account
from google.cloud import bigquery

import sqlalchemy
from sqlalchemy import create_engine

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")

def bq_conn(sql):
    '''
    Execute sql on big query
    '''

    try:
        credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
        client = bigquery.Client(GOOGLE_PROJECT, credentials)
        df = client.query(sql, project=GOOGLE_PROJECT).to_dataframe()

        return df

    except Exception as e:
        print(e)
        return 'error running query'

def get_simulation_jobs():

    sql = f"""
        select *
        from `{BQ_DATASET}.staging_eplus_job` i
        order by created_at desc
    """

    df = bq_conn(sql)
    return df

def get_simulation_data(simid, target, zones=[]):
    '''
    Get simulation data from big query
    '''

    simid_list = "'"+"','".join(simid)+"'"
    if zones==[]:
        zones_list = ''
    else:
        zl = "'"+"','".join(zones)+"'"
        zones_list = 'and loc in (' + zl + ')'

    sql = f"""
        select i.*, o.loc, o.{target}
        from `{BQ_DATASET}.staging_simulation_input` i
        inner join `{BQ_DATASET}.staging_simulation_output` o
        on i.id  = o.input_id
        where i.job_id in ({simid_list})
        {zones_list}
    """

    df = bq_conn(sql)
    df = df.drop(columns = ['metadata'])
    df = df.rename(columns={"id": "sample_id"})
    return df

def get_simulation_jobs():
    '''
    Get all of the simulation jobs from big query
    '''

    sql = f"""
        select *
        from `{BQ_DATASET}.staging_eplus_job` i
        order by created_at desc
    """

    df = bq_conn(sql)
    return df

def get_training_data(db_config, schema_table, target):

    DB_DRIVER=db_config['DB_DRIVER']
    DB_USER=db_config['DB_USER']
    DB_PASSWORD=db_config['DB_PASSWORD']
    DB_HOST=db_config['DB_HOST']
    DB_PORT=db_config['DB_PORT']
    DB_NAME=db_config['DB_NAME']

    engine = create_engine(f'{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    conn = engine.connect()

    query_string = f"""
        SELECT * 
        FROM {schema_table}
    """

    df = pd.read_sql(query_string, conn)
    conn.close()

    df[target] = df[config['target']].astype(float)
    return df