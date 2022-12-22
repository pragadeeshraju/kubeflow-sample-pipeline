from typing import NamedTuple
from tensorflow import keras
from minio import Minio
import numpy as np
import json
from collections import namedtuple

def get_data_batch() -> NamedTuple('Outputs', [('datapoints_training', float),('datapoints_test', float),('dataset_version', str)]):
    """
    Function to get dataset and load it to minio bucket
    """
    print("getting data")

    minio_client = Minio(
        "minio-service.kubeflow.svc.cluster.local:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    minio_client.fget_object(minio_bucket,"mnist.npz","/tmp/mnist.npz")
    
    def load_data():
        with np.load("/tmp/mnist.npz", allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)
    
    # Get MNIST data directly from library
    (x_train, y_train), (x_test, y_test) = load_data()

    # save to numpy file, store in Minio
    np.save("/tmp/x_train.npy",x_train)
    minio_client.fput_object(minio_bucket,"mnistdocker/x_train","/tmp/x_train.npy")

    np.save("/tmp/y_train.npy",y_train)
    minio_client.fput_object(minio_bucket,"mnistdocker/y_train","/tmp/y_train.npy")

    np.save("/tmp/x_test.npy",x_test)
    minio_client.fput_object(minio_bucket,"mnistdocker/x_test","/tmp/x_test.npy")

    np.save("/tmp/y_test.npy",y_test)
    minio_client.fput_object(minio_bucket,"mnistdocker/y_test","/tmp/y_test.npy")
    
    dataset_version = "1.0"
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    divmod_output = namedtuple('Outputs', ['datapoints_training', 'datapoints_test', 'dataset_version'])
    return [float(x_train.shape[0]),float(x_test.shape[0]),dataset_version]

if __name__ == '__main__':
    get_data_batch()