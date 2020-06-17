from __future__ import print_function
import json
import requests
from datetime import datetime
from time import gmtime, strftime

# airflow operators
import airflow
from airflow.models import DAG
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

# airflow sagemaker operators
from airflow.contrib.operators.sagemaker_training_operator \
    import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_tuning_operator \
    import SageMakerTuningOperator
from airflow.contrib.operators.sagemaker_transform_operator \
    import SageMakerTransformOperator
from airflow.contrib.operators.sagemaker_model_operator \
    import SageMakerModelOperator
from airflow.contrib.operators.sagemaker_endpoint_operator \
    import SageMakerEndpointOperator

from airflow.contrib.hooks.aws_hook import AwsHook

# sagemaker sdk
import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel

# airflow sagemaker configuration
from sagemaker.workflow.airflow import training_config
from sagemaker.workflow.airflow import tuning_config
from sagemaker.workflow.airflow import transform_config_from_estimator
from sagemaker.workflow.airflow import model_config
from sagemaker.workflow.airflow import transform_config
from sagemaker.workflow.airflow import deploy_config

# ml workflow specific

from jay_pipeline.sm_proc_job import sm_proc_job
from jay_pipeline.inference_pipeline_ep import inference_pipeline_ep

import jay_pipeline.config as cfg
import jay_pipeline.schema_utils

# =============================================================================
# functions
# =============================================================================


def is_hpo_enabled():
    """check if hyper-parameter optimization is enabled in the config
    """
    hpo_enabled = False
    if "job_level" in config and \
            "run_hyperparameter_opt" in config["job_level"]:
        run_hpo_config = config["job_level"]["run_hyperparameter_opt"]
        if run_hpo_config.lower() == "yes":
            hpo_enabled = True
    return hpo_enabled


def get_sagemaker_role_arn(role_name, region_name):
    iam = boto3.client('iam', region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]

def create_def_input_data_channels(s3_train_data, s3_validation_data):    
    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                        content_type='text/csv', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                             content_type='text/csv', s3_data_type='S3Prefix')
    return {'train': train_data, 'validation': validation_data}

# =============================================================================
# setting up training, tuning and transform configuration
# =============================================================================


# read config file
config = cfg.config

# set configuration for tasks
hook = AwsHook(aws_conn_id='airflow-sagemaker')
region = config["job_level"]["region_name"]
sess = hook.get_session(region_name=region)
role = get_sagemaker_role_arn(
    config["train_model"]["sagemaker_role"],
    sess.region_name)
hpo_enabled = is_hpo_enabled()

# create XGB estimator
xgb_container = get_image_uri(sess.region_name, 'xgboost', repo_version="0.90-1")

xgb_estimator = Estimator(
    image_name=xgb_container,
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    **config["train_model"]["estimator_config"]
)

# train_config specifies SageMaker training configuration
data_channels = create_def_input_data_channels(config['train_model']['inputs']['train'],config['train_model']['inputs']['validation'])

train_config = training_config(
    estimator=xgb_estimator,
    inputs=data_channels)

# =============================================================================
# define airflow DAG and tasks
# =============================================================================
# define airflow DAG
args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    'ml-pipeline',
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
    user_defined_filters={'tojson': lambda s: json.JSONEncoder().encode(s)}
)

# Set the tasks in the DAG

# Dummy start operator
init = DummyOperator(
    task_id='start',
    dag=dag
)

# SageMaker processing job task
sm_proc_job_task = PythonOperator(
    task_id='sm_proc_job',
    dag=dag,
    provide_context=True,
    python_callable=sm_proc_job,
    op_kwargs={'role': role, 'sess': sess})

# Train xgboost model task
train_model_task = SageMakerTrainingOperator(
    task_id='xgboost_model_training',
    dag=dag,
    config=train_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30
)

cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)

init.set_downstream(sm_proc_job_task)
sm_proc_job_task.set_downstream(train_model_task)
train_model_task.set_downstream(cleanup_task)
