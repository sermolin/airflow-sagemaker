#sm_proc_job.py
import sagemaker
from time import gmtime, strftime
import boto3
import os
import sys

def sm_proc_job (role, sess):

  #sagemaker_session = sagemaker.Session()
  #role = sagemaker.get_execution_role()
  #bucket = sagemaker_session.default_bucket()
  bucket = 'airflow-sagemaker-2'

  timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

  prefix = 'sagemaker/spark-preprocess-demo/' + timestamp_prefix
  input_prefix = 'sagemaker/spark-preprocess-demo/input/raw/abalone'
  input_preprocessed_prefix = prefix + '/input/preprocessed/abalone'
  model_prefix = prefix + '/model'


  from sagemaker.processing import ScriptProcessor, ProcessingInput

  #account_id = boto3.client('sts').get_caller_identity().get('Account')
  #region = boto3.session.Session().region_name
  #ecr_repository = 'sagemaker-spark-example'
  #tag = ':latest'
  #uri_suffix = 'amazonaws.com'
  #spark_repository_uri = '{}.dkr.ecr.{}.{}/{}'.format(account_id, region, uri_suffix, ecr_repository + tag)
  #prebuilt container
  spark_repository_uri = '328296961357.dkr.ecr.us-east-1.amazonaws.com/sagemaker-spark-example:latest'

  # Create ECR repository and push docker image
  spark_processor = ScriptProcessor(base_job_name='spark-preprocessor',
                                    image_uri=spark_repository_uri,
                                    command=['/opt/program/submit'],
                                    role=role,
                                    sagemaker_session=sagemaker.Session(sess),
                                    instance_count=2,
                                    instance_type='ml.r5.xlarge',
                                    max_runtime_in_seconds=1200,
                                    env={'mode': 'python'})

  spark_processor.run(code='s3://airflow-sagemaker-2/smprocpreprocess.py', arguments=['s3_input_bucket', bucket, 's3_input_key_prefix', input_prefix, 's3_output_bucket', bucket, 's3_output_key_prefix', input_preprocessed_prefix], logs=False)
