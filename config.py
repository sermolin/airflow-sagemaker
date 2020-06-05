from datetime import datetime
from sagemaker.tuner import ContinuousParameter

config = {}

config["job_level"] = {
    "region_name": "us-east-1",
    "run_hyperparameter_opt": "no"
}

config["preprocess_data"] = {
    "s3_in_url": "s3://amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz",
    "s3_out_bucket": "airflow-sagemaker-2",  # replace
    "s3_out_prefix": "preprocess/",
    "delimiter": "\t"
}

config["prepare_data"] = {
    "s3_in_bucket": "airflow-sagemaker-2",  # replace
    "s3_in_prefix": "preprocess/",
    "s3_out_bucket": "airflow-sagemaker-2",  # replace
    "s3_out_prefix": "prepare/",
    "delimiter": "\t"
}

config["train_model"] = {
    "sagemaker_role": "AirflowSageMakerExecutionRole",
    "estimator_config": {
        "train_instance_count": 1,
        "train_instance_type": "ml.m4.xlarge",
        "train_volume_size": 20,
        "train_max_run": 3600,
        "output_path": "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/xgboost_model/output/",  # replace
        "base_job_name": "c1-xgb-airflow",
        "hyperparameters": {
            "objective" : "reg:linear",
            "eta" : ".2",
            "max_depth" : "5",
            "num_round" : "10",
            "subsample" : "0.7",
            "silent"    : "0",
            "min_child_weight" : "6"   
        }
    },
    "inputs": {
        "train": "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-05-00-56-24/input/preprocessed/abalone/train/part-00000",
        "validation": "s3://airflow-sagemaker-2/sagemaker/spark-preprocess-demo/2020-06-05-00-56-24/input/preprocessed/abalone/validation/part-00000"  # replace
    }
}

config["tune_model"] = {
    "tuner_config": {
        "objective_metric_name": "test:rmse",
        "objective_type": "Minimize",
        "hyperparameter_ranges": {
            "factors_lr": ContinuousParameter(0.0001, 0.2),
            "factors_init_sigma": ContinuousParameter(0.0001, 1)
        },
        "max_jobs": 20,
        "max_parallel_jobs": 2,
        "base_tuning_job_name": "hpo-recommender"
    },
    "inputs": {
        "train": "s3://airflow-sagemaker-2/prepare/train/train.protobuf",  # replace
        "test": "s3://airflow-sagemaker-2/prepare/validate/validate.protobuf"  # replace
    }
}

config["batch_transform"] = {
    "transform_config": {
        "instance_count": 1,
        "instance_type": "ml.c4.xlarge",
        "data": "s3://airflow-sagemaker-2/prepare/test/",
        "data_type": "S3Prefix",
        "content_type": "application/x-recordio-protobuf",
        "strategy": "MultiRecord",
        "output_path": "s3://airflow-sagemaker-2/transform/"
    }
}
