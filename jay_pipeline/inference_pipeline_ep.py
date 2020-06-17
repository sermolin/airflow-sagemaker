import sagemaker
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.model import Model
from time import gmtime, strftime
import boto3
import os
import sys


def inference_pipeline_ep(role, sess):

    #sagemaker_session = sagemaker.Session()
    #role = sagemaker.get_execution_role()
    #bucket = sagemaker_session.default_bucket()
    bucket = 'airflow-jeprk-sagemaker'

    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    prefix = 'sagemaker/spark-preprocess-demo/' + timestamp_prefix
    input_prefix = 'sagemaker/spark-preprocess-demo/input/raw/abalone'
    input_preprocessed_prefix = prefix + '/input/preprocessed/abalone'
    model_prefix = prefix + '/model'
    xgb_container = get_image_uri(
        sess.region_name, 'xgboost', repo_version="0.90-1")
    spark_repository_uri = '328296961357.dkr.ecr.us-east-1.amazonaws.com/sagemaker-spark-example:latest'
    s3_uri_model_location = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/xgboost_model/c1-xgb-airflow-2020-06-05-20-34-41-923/output/model.tar.gz"
    s3_sparkml_data = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-13-00-22-56/mleap-model/model.tar.gz"

    model_name = 'xgb-model-abalone'
    schema_json = schema_utils.abalone_schema()

    # REAL-TIME INFERENCE
    # passing the schema defined above by using an environment variable that sagemaker-sparkml-serving understands
    #sparkml_model = SparkMLModel(model_data=s3_sparkml_data, env={'SAGEMAKER_SPARKML_SCHEMA' : schema_json})
    sparkml_model = SparkMLModel(model_data=s3_sparkml_data,  role=role, sagemaker_session=sagemaker.session.Session(
        sess), env={'SAGEMAKER_SPARKML_SCHEMA': schema_json})
    xgb_model = Model(model_data=s3_uri_model_location, role=role,
                      sagemaker_session=sagemaker.session.Session(sess), image=xgb_container)
    pipline_model_name = 'inference-pipeline-' + timestamp_prefix
    sm_model = PipelineModel(name=pipline_model_name,
                             role=role,
                             sagemaker_session=sagemaker.session.Session(sess),
                             models=[sparkml_model, xgb_model])

    endpoint_name = 'inference-pipeline-ep-' + timestamp_prefix
    sm_model.deploy(initial_instance_count=1,
                    instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
