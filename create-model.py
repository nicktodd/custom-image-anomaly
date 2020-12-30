import boto3
import sagemaker
from sagemaker import get_execution_role

client = boto3.client('sts')
account = client.get_caller_identity()['Account']

role = "arn:aws:iam::963778699255:role/service-role/AmazonSageMaker-ExecutionRole-20190723T151113" #get_execution_role()

region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
bucket = sagemaker_session.default_bucket()
prefix = 'anomaly-ml-image'
print(region)
print(role)
print(bucket)

docker_image_name = account + ".dkr.ecr.eu-west-1.amazonaws.com/anomalyimage:latest"

sess = sagemaker.session.Session()
anomaly_detection = sagemaker.estimator.Estimator(image_uri=docker_image_name,
                                    role=role,
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    base_job_name="anomaly-detection",
                                    sagemaker_session=sess)

anomaly_detection.fit()

model_data_s3_path = anomaly_detection.model_data
print("S3 path to model is " + model_data_s3_path)


# Set up the Cloudformation parameters
parameter_file_data = {
    
        "Parameters" : {
            "ModelName" : "CustomMLModel",
            "ModelDataUrl" : model_data_s3_path,
            "TrainingImage": docker_image_name,
            "InstanceType" : "ml.t2.xlarge",
            "InstanceCount": str(1),
            "RoleArn": role
        }
    
}
print("using this parameter file")
print(parameter_file_data)

import json
with open('cloudformation_parameters.json', 'w') as outfile:    
        json.dump(parameter_file_data, outfile)
