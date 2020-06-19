from time import gmtime, strftime
import time

timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
bucket = 'airflow-sm-jeprk'

config = {}

config["job_level"] = {
    "region_name": "us-east-1",
    "run_hyperparameter_opt": "no"
}

config["timestamp"] = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

config["bucket"] = bucket

config["train_model"] = {
    "sagemaker_role": "AirflowSageMakerExecutionRole",
    "estimator_config": {
        "train_instance_count": 1,
        "train_instance_type": "ml.m4.xlarge",
        "train_volume_size": 20,
        "train_max_run": 3600,
        "output_path": "s3://"+bucket+"/sagemaker/spark-preprocess/model/xgboost",
        "base_job_name": "training-job-",
        "hyperparameters": {
            "objective": "reg:linear",
            "eta": ".2",
            "max_depth": "5",
            "num_round": "10",
            "subsample": "0.7",
            "silent": "0",
            "min_child_weight": "6"
        }
    },
    "inputs": {
        "train": "s3://"+bucket+"/sagemaker/spark-preprocess/input/preprocessed/abalone/"+timestamp_prefix+"train/part-00000",
        "validation": "s3://"+bucket+"/sagemaker/spark-preprocess/input/preprocessed/"+timestamp_prefix+"/abalone/validation/part-00000"  # replace
    }
}

config["inference_pipeline"] = {
    "inputs": {
        "spark_model": "s3://"+bucket+"/sagemaker/spark-preprocess/model/spark/"+timestamp_prefix+"model.tar.gz"
    }
}

config["batch_transform"] = {
    "transform_config": {
        "instance_count": 1,
        "instance_type": "ml.c4.xlarge",
        "data": "s3://"+bucket+"/prepare/test/",
        "data_type": "S3Prefix",
        "content_type": "application/x-recordio-protobuf",
        "strategy": "MultiRecord",
        "output_path": "s3://"+bucket+"/transform/"
    },
    "inputs": "s3://"+bucket+"/sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv",
    "input_filter": "$[1:]",
    "model_name": "inference-pipeline-spark-xgboost"+timestamp_prefix
}
