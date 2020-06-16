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

from jay_pipeline import sm_proc_job
import inference_pipeline_ep

import config_test as cfg
import schema_utils

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

# =============================================================================
# define airflow DAG and tasks
# =============================================================================
# define airflow DAG
args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='sagemaker-ml-pipeline-proc',
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
    user_defined_filters={'tojson': lambda s: json.JSONEncoder().encode(s)}
)

# set the tasks in the DAG

# dummy operator
init = DummyOperator(
    task_id='start',
    dag=dag
)

sm_proc_job_task = PythonOperator(
    task_id='sm_proc_job',
    dag=dag,
    provide_context=True,
    python_callable=sm_proc_job.sm_proc_job,
    op_kwargs={'role': role, 'sess': sess})


create_endpoint_task = PythonOperator(
    task_id='create_endpoint',
    dag=dag,
    provide_context=False,
    python_callable=inference_pipeline_ep.inference_pipeline_ep,
    op_kwargs={'role': role, 'sess': sess})

cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)

init.set_downstream(sm_proc_job_task)
