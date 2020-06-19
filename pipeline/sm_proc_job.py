# sm_proc_job.py
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput
import boto3
import os
import sys

def sm_proc_job(role, sess, timestamp, **context):

    bucket = "airflow-sagemaker-jeprk"

    prefix = "sagemaker/spark-preprocess-demo/"
    input_prefix = 'sagemaker/spark-preprocess-demo/input/raw/abalone'
    input_preprocessed_prefix = prefix + "/input/preprocessed/"+timestamp+"abalone"
    model_prefix = prefix + "model/spark/"+timestamp

    spark_repository_uri = "885332847160.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark"

    spark_processor = ScriptProcessor(base_job_name="spark-preprocessor",
                                      image_uri=spark_repository_uri,
                                      command=["/opt/program/submit"],
                                      role=role,
                                      sagemaker_session=sagemaker.Session(
                                          sess),
                                      instance_count=2,
                                      instance_type="ml.r5.xlarge",
                                      max_runtime_in_seconds=1200,
                                      env={"mode": "python"})

    spark_processor.run(code="s3://airflow-sagemaker-jeprk/code/smprocpreprocess.py", arguments=["s3_input_bucket", bucket, "s3_input_key_prefix", input_prefix,
                                                                                                 "s3_output_bucket", bucket, "s3_output_key_prefix", input_preprocessed_prefix, "s3_model_bucket", bucket, "s3_model_prefix", model_prefix], logs=True)
