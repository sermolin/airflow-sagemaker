from datetime import datetime

config = {}

config["job_level"] = {
    "region_name": "us-west-2",
    "run_hyperparameter_opt": "no"
}

config["train_model"] = {
    "sagemaker_role": "AirflowSageMakerExecutionRole",
    "estimator_config": {
        "train_instance_count": 1,
        "train_instance_type": "ml.m4.xlarge",
        "train_volume_size": 20,
        "train_max_run": 3600,
        "output_path": "s3://airflow-sagemaker-jeprk/sagemaker/spark-preprocess-demo/model/xgboost", #replace
        "base_job_name": "xgboost-training",
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
        "train": "s3://airflow-sagemaker-jeprk/sagemaker/spark-preprocess-demo/input/preprocessed/abalone/train/part-00000",
        "validation": "s3://airflow-sagemaker-jeprk/sagemaker/spark-preprocess-demo/input/preprocessed/abalone/validation/part-00000"  # replace
    }
}

config["inference_pipeline"] = {
    "inputs": {
        "spark_model": "s3://airflow-sagemaker-jeprk/sagemaker/spark-preprocess-demo/model/spark/model.tar.gz"
    }
}

config["batch_transform"] = {
    "transform_config": {
        "instance_count": 1,
        "instance_type": "ml.c4.xlarge",
        "data": "s3://airflow-sagemaker-jeprk/prepare/test/",
        "data_type": "S3Prefix",
        "content_type": "application/x-recordio-protobuf",
        "strategy": "MultiRecord",
        "output_path": "s3://airflow-sagemaker-jeprk/transform/"
    },
    "inputs": "s3://airflow-sagemaker-jeprk/sagemaker/spark-preprocess-demo/batch_input/batch_input_abalone.csv",
    "model_name": 'inference-pipeline-spark-xgboost'
}
