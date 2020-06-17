import sagemaker
     from sagemaker.amazon.amazon_estimator import get_image_uri
def train_xgboost_model(role, sess, s3_train_data, s3_validation_data, s3_output_location):
     training_image = get_image_uri(sess.boto_region_name, 'xgboost', repo_version="0.90-1")

     xgb_model = sagemaker.estimator.Estimator(training_image,
                                               role, 
                                               train_instance_count=1, 
                                               train_instance_type='ml.m4.xlarge',
                                               train_volume_size = 20,
                                               train_max_run = 3600,
                                               input_mode= 'File',
                                               output_path=s3_output_location,
                                               sagemaker_session=sess)

     xgb_model.set_hyperparameters(objective = "reg:linear",
                                   eta = .2,
                                   gamma = 4,
                                   max_depth = 5,
                                   num_round = 10,
                                   subsample = 0.7,
                                   silent = 0,
                                   min_child_weight = 6)

     train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                             content_type='text/csv', s3_data_type='S3Prefix')
     validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                  content_type='text/csv', s3_data_type='S3Prefix')

     data_channels = {'train': train_data, 'validation': validation_data}
     xgb_model.fit(inputs=data_channels, logs=True)