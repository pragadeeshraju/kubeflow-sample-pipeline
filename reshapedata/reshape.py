def reshape_data():
    """
    Reshape the data for model building
    """
    print("reshaping data")
    
    from minio import Minio
    import numpy as np

    minio_client = Minio(
        "10.0.102.158:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    # load data from minio
    minio_client.fget_object(minio_bucket,"mnistdocker/x_train","/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")
    
    minio_client.fget_object(minio_bucket,"mnistdocker/x_test","/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")
    
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    
    # save data from minio
    np.save("/tmp/x_train.npy",x_train)
    minio_client.fput_object(minio_bucket,"mnistdocker/x_train","/tmp/x_train.npy")
    
    np.save("/tmp/x_test.npy",x_test)
    minio_client.fput_object(minio_bucket,"mnistdocker/x_test","/tmp/x_test.npy")

if __name__ == '__main__':
    reshape_data()

