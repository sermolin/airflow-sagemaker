from airflow.models import Variable
import time
from time import gmtime, strftime
import os
import subprocess
import boto3

s3 = boto3.client('s3')


def find_files(file_name):
    command = ['locate', file_name]

    output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    output = output.decode()

    search_results = output.split('\n')

    return search_results


def start(bucket):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)
    find_files('smprocpreprocess.py')
    # s3.put_object(
    #     Bucket=bucket, Key='sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv', Body=abalone.csv)
    # s3.put_object(Bucket=bucket, Key='code/smprocpreprocess.py',
    #               Body=smprocpreprocess.py)
    time.sleep(10)
