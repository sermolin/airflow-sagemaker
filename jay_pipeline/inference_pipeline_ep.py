import sagemaker
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.model import Model
from time import gmtime, strftime
import boto3
import os
import sys
from sagemaker.amazon.amazon_estimator import get_image_uri

def inference_pipeline_ep(role, sess, xgb_model_uri, spark_model_uri):
    s3_sparkml_data_uri = spark_model_uri
    s3_xgboost_model_uri = xbg_model_uri

    xgb_container = get_image_uri(
        sess.region_name, 'xgboost', repo_version="0.90-1")

    schema_json = schema_utils.abalone_schema()

    sparkml_model = SparkMLModel(model_data=s3_sparkml_data_uri,  role=role, sagemaker_session=sagemaker.session.Session(
        sess), env={'SAGEMAKER_SPARKML_SCHEMA': schema_json})

    xgb_model = Model(model_data=s3_xgboost_model_uri, role=role,
                      sagemaker_session=sagemaker.session.Session(sess), image=xgb_container)

    pipeline_model_name = 'inference-pipeline-spark-xgboost'

    sm_model = PipelineModel(name=pipeline_model_name,
                             role=role,
                             sagemaker_session=sagemaker.session.Session(sess),
                             models=[sparkml_model, xgb_model])

    endpoint_name = 'inference-pipeline-endpoint'
    
    sm_model.deploy(initial_instance_count=1,
                    instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
