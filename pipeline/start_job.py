from airflow.models import Variable
import time
from time import gmtime, strftime
import os
import subprocess
import boto3

s3 = boto3.client('s3')


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def start(bucket):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)
    find_all('smprocpreprocess.py', '/')
    # s3.put_object(
    #     Bucket=bucket, Key='sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv', Body=abalone.csv)
    # s3.put_object(Bucket=bucket, Key='code/smprocpreprocess.py',
    #               Body=smprocpreprocess.py)
    time.sleep(10)
