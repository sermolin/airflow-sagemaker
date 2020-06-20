from airflow.models import Variable

timestamp = ""

try:
    timestamp = Variable.get("timestamp")
except:
    pass

bucket = 'airflow-sm-jeprk'

config = {}

config["job_level"] = {
    "region_name": "us-east-1",
    "run_hyperparameter_opt": "no"
}

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
        "train": "s3://"+bucket+"/sagemaker/spark-preprocess/inputs/preprocessed/abalone/"+timestamp+"/train/part-00000",
        "validation": "s3://"+bucket+"/sagemaker/spark-preprocess/inputs/preprocessed/abalone/"+timestamp+"/validation/part-00000"  # replace
    }
}

config["inference_pipeline"] = {
    "inputs": {
        "spark_model": "s3://"+bucket+"/sagemaker/spark-preprocess/model/spark/"+timestamp+"/model.tar.gz"
    }
}

config["batch_transform"] = {
    "transformer_config": {
        "instance_count": 1,
        "instance_type": "ml.c4.xlarge",
        "output_path": "s3://" + bucket + "/sagemaker/spark-preprocess/batch_output/xgb-transform/" +timestamp
    },
    "transform_config": {
        "data": "s3://"+bucket+"/sagemaker/spark-preprocess/inputs/raw/abalone/abalone.csv",
        "data_type": "S3Prefix",
        "content_type": "text/csv",
        "split_type": "Line",
        "input_filter": "$[:-1]",
        "job_name": "xgb-transform-job-"+timestamp
    },
    "model_name": "inference-pipeline-spark-xgboost-"+timestamp
}
