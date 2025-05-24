from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path='s3://sagemaker-us-east-1-123456789012/tensorboard',
        container_local_output_path='/opt/ml/output/tensorboard'
    )

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='training',
        role='arn:aws:iam::123456789012:role/SageMakerRole',
        framework_version='2.5.1',
        py_version='py311',
        instance_type='ml.g5.xlarge'
        instance_count=1,
        hyperparameters={
            'batch_size': 32,
            'epochs': 25
        },
        tensorboard_config=tensorboard_config,
    )

    # Start training
    estimator.fit({
        'training': 's3://sagemaker-us-east-1-123456789012/training',
        'validation': 's3://sagemaker-us-east-1-123456789012/validation',
        'test': 's3://sagemaker-us-east-1-123456789012/test'
    })

    print('Training completed!')

if __name__ == "__main__":
    start_training()