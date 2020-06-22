from airflow.models import Variable
import time
from time import gmtime, strftime
import boto3
import os
import config as cfg

s3 = boto3.client('s3')


def start(bucket):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)
    print(os.getcwd())
    # s3.upload_file(
    #     Bucket=bucket, Key='sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv', Filename='./abalone.csv')
    # s3.upload_file(Bucket=bucket, Key='code/smprocpreprocess.py',
    #                Filename='./smprocpreprocess.py')
    time.sleep(10)
