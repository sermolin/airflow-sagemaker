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
#from sagemaker.tuner import HyperparameterTuner
#from sagemaker.model import FrameworkModel
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
#from pipeline import prepare, preprocess
#from pipeline import sm_proc_job, sm_proc_preprocess
#
from pipeline import sm_proc_job
import inference_pipeline_ep
#

import config as cfg
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

xgb_container = get_image_uri(sess.region_name, 'xgboost', repo_version="0.90-1")
timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
hpo_enabled = is_hpo_enabled()

# create XGB estimator
xgb_estimator = Estimator(
    image_name=xgb_container,
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    **config["train_model"]["estimator_config"]
)

# train_config specifies SageMaker training configuration
## Original
##train_config = training_config(
##    estimator=xgb_estimator,
##    inputs=config["train_model"]["inputs"])

s3_train_data = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-05-00-56-24/input/preprocessed/abalone/train/part-00000"
s3_validation_data = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-05-00-56-24/input/preprocessed/abalone/validation/part-00000"
s3_test_data = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-05-00-56-24/input/preprocessed/abalone/test/preproc-test.csv"
s3_uri_model_location = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/xgboost_model/c1-xgb-airflow-2020-06-05-20-34-41-923/output/model.tar.gz"
s3_sparkml_data = "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-13-00-22-56/mleap-model/model.tar.gz"
#s3_sparkml_data = "s3://sagemaker-us-east-1-328296961357/sagemaker/spark-preprocess-demo/2020-06-12-18-31-52/mleap-model/model.tar.gz"

train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                        content_type='text/csv', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                             content_type='text/csv', s3_data_type='S3Prefix')

test_data = sagemaker.session.s3_input(s3_test_data, distribution='FullyReplicated', 
                             content_type='text/csv', s3_data_type='S3Prefix')

data_channels = {'train': train_data, 'validation': validation_data}

train_config = training_config(
    estimator=xgb_estimator,
    inputs=data_channels)

model_name = 'xgb-model-abalone-spark-1'
schema_json = schema_utils.abalone_schema()

# MODEL COMPILATION
xgb_model = Model(
    model_data = s3_uri_model_location,
    image = xgb_container,
    role = role,
    name = model_name,
    sagemaker_session = sagemaker.session.Session(sess)
    )

#create model config
xgb_model_config = model_config(
    instance_type = 'ml.c5.xlarge',
    model = xgb_model,
    role = role,
    image = xgb_container
    )

# BATCH INFERENCE

xgb_transformer = Transformer(
    model_name = model_name,
    instance_count = 1,
    instance_type = 'ml.c5.xlarge',
    sagemaker_session = sagemaker.session.Session(sess)
    )

transform_config = transform_config (
    transformer = xgb_transformer,
    job_name = 'xgb-tranform-job-' + timestamp_prefix,
    data = s3_test_data,
    content_type='text/csv',
    split_type='Line',
    #input_filter='$[1:]',
    data_type = 'S3Prefix'
    )

# REAL-TIME INFERENCE
# passing the schema defined above by using an environment variable that sagemaker-sparkml-serving understands
#sparkml_model = SparkMLModel(model_data=s3_sparkml_data, env={'SAGEMAKER_SPARKML_SCHEMA' : schema_json})
#sparkml_model = SparkMLModel(model_data=s3_sparkml_data,  role=role, sagemaker_session = sagemaker.session.Session(sess), env={'SAGEMAKER_SPARKML_SCHEMA' : schema_json})
#xgb_model = Model(model_data=s3_uri_model_location, image=training_image) #if compiling the model 1st time
#pipline_model_name = 'inference-pipeline-' + timestamp_prefix
#sm_model = PipelineModel(name=pipline_model_name, 
#    role=role, 
#    sagemaker_session = sagemaker.session.Session(sess), 
#    models=[sparkml_model, xgb_model])

#endpoint_name = 'inference-pipeline-ep-' + timestamp_prefix
#sm_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

#create model config
#pipeline_model_config = model_config(
#    instance_type = 'ml.c5.xlarge',
#    model = sm_model,
#    role = role
#    )

#pipeline_deploy_config = deploy_config(
#    model = sm_model,
#    model = sparkml_model,
#    initial_instance_count = 1,
#    instance_type = 'ml.c5.xlarge',
#    endpoint_name = endpoint_name
#    )

# =============================================================================
# define airflow DAG and tasks
# =============================================================================

# define airflow DAG

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='sagemaker-ml-pipeline-proc-5',
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

##sm_proc_job_task = PythonOperator(
##    task_id='sm_proc_job',
##    dag=dag,
##    provide_context=False,
##    python_callable=sm_proc_job.sm_proc_job,
##    op_kwargs= {'role': role, 'sess': sess})

# launch sagemaker training job and wait until it completes
##train_model_task = SageMakerTrainingOperator(
##    task_id='model_training',
##    dag=dag,
##    config=train_config,
##    aws_conn_id='airflow-sagemaker',
##    wait_for_completion=True,
##    check_interval=30
##)

##compile_model_task = SageMakerModelOperator(
##    task_id='compile_model',
##    dag=dag,
##    config=xgb_model_config,
##    aws_conn_id='airflow-sagemaker',
##    wait_for_completion=True,
##    check_interval=30
##)

#launch sagemaker batch transform job and wait until it completes
##batch_transform_task = SageMakerTransformOperator(
##    task_id='batch_predicting',
##    dag=dag,
##    config=transform_config,
##    aws_conn_id='airflow-sagemaker',
##    wait_for_completion=True,
##    check_interval=30
##)

#compile_pipeline_model_task = SageMakerModelOperator(
#    task_id='compile_pipeline_model',
#    dag=dag,
#    config=pipeline_model_config,
#    aws_conn_id='airflow-sagemaker',
#    wait_for_completion=True,
#    check_interval=30
#)

#create_endpoint_task = SageMakerEndpointOperator(
#    task_id='create_endpoint',
#    dag=dag,
#    config=pipeline_deploy_config,
#    aws_conn_id='airflow-sagemaker',
#    wait_for_completion=True,
#    check_interval=30)

create_endpoint_task = PythonOperator(
    task_id='create_endpoint',
    dag=dag,
    provide_context=False,
    python_callable=inference_pipeline_ep.inference_pipeline_ep,
    op_kwargs= {'role': role, 'sess': sess})

cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)

# set the dependencies between tasks
init.set_downstream(create_endpoint_task)
create_endpoint_task.set_downstream(cleanup_task)

#init.set_downstream(compile_model_task)
#compile_model_task.set_downstream(batch_transform_task)
#batch_transform_task.set_downstream(cleanup_task)

#init.set_downstream(sm_proc_job_task)
#sm_proc_job_task.set_downstream(train_model_task)
#train_model_task.set_downstream(cleanup_task)
#init.set_downstream(sm_proc_job_task)

##sm_proc_job_task.set_downstream(cleanup_task)
#sm_proc_preprocess_task.set_downstream(sm_proc_job_task)
#sm_proc_job_task.set_downstream(preprocess_task)
#init.set_downstream(preprocess_task)
#preprocess_task.set_downstream(prepare_task)
#prepare_task.set_downstream(train_model_task)

#batch_transform_task.set_downstream(cleanup_task)

#init.set_downstream(preprocess_task)
#preprocess_task.set_downstream(prepare_task)
#prepare_task.set_downstream(branching)
#branching.set_downstream(tune_model_task)
#branching.set_downstream(train_model_task)
#tune_model_task.set_downstream(batch_transform_task)
#train_model_task.set_downstream(batch_transform_task)
#batch_transform_task.set_downstream(cleanup_task)

#TODO:
# - pass output data directory from preprocessing to training (currently, s3_train_data and s3_validation_data are hard-coded)
# - pass data_channels via config.py

