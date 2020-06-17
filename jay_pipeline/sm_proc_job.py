# sm_proc_job.py
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput
from time import gmtime, strftime
import boto3
import os
import sys

def sm_proc_job(role, sess, **context):

    bucket = 'airflow-sagemaker-jeprk'

    prefix = 'sagemaker/spark-preprocess-demo/'
    input_prefix = 'sagemaker/spark-preprocess-demo/input/raw/abalone'
    input_preprocessed_prefix = prefix + '/input/preprocessed/abalone'
    model_prefix = prefix + 'model/spark'

    spark_repository_uri = '885332847160.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark'

    # Create ECR repository and push docker image
    spark_processor = ScriptProcessor(base_job_name='spark-preprocessor',
                                      image_uri=spark_repository_uri,
                                      command=['/opt/program/submit'],
                                      role=role,
                                      sagemaker_session=sagemaker.Session(
                                          sess),
                                      instance_count=2,
                                      instance_type='ml.r5.xlarge',
                                      max_runtime_in_seconds=1200,
                                      env={'mode': 'python'})

    spark_processor.run(code='s3://airflow-sagemaker-jeprk/code/smprocpreprocess.py', arguments=['s3_input_bucket', bucket, 's3_input_key_prefix', input_prefix,
                                                                                                 's3_output_bucket', bucket, 's3_output_key_prefix', input_preprocessed_prefix, 's3_model_bucket', bucket, 's3_model_prefix', model_prefix], logs=True)
    s3_training_path = 's3://' + bucket + \
        input_preprocessed_prefix + '/train/part-00000'
    s3_validation_path = 's3://' + bucket + \
        input_preprocessed_prefix + '/validation/part-00000'
    s3_model_path = 's3://' + bucket + model_prefix + '/model.tar.gz'
    return s3_training_path, s3_validation_path, s3_model_path
