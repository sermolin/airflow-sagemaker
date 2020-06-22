from airflow.models import DAG, Variable
from time import gmtime, strftime
import boto3
import config as cfg

s3 = boto3.client('s3')


def start(bucket):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)
    s3.put_object(
        Bucket=bucket, Key='sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv', Body='/abalone.csv')
    s3.put_object(Bucket=bucket, key='code/smprocpreprocess.py',
                  Body='/smprocpreprocess.py')
