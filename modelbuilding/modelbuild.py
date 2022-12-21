from tensorflow import keras
import tensorflow as tf
from minio import Minio
import numpy as np
import pandas as pd
import json
import os
import glob
from collections import namedtuple
from typing import NamedTuple

def model_building(
    no_epochs:int = 1,
    optimizer: str = "adam"
) -> NamedTuple('Output', [('mlpipeline_ui_metadata', 'UI_metadata'),('mlpipeline_metrics', 'Metrics')]):
    """
    Build the model with Keras API
    Export model parameters
    """

    minio_client = Minio(
        "10.0.102.158:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax')) #output are 10 classes, numbers from 0-9

    #show model summary - how it looks
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    metric_model_summary = "\n".join(stringlist)
    
    #compile the model - we want to have a binary outcome
    model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    
    minio_client.fget_object(minio_bucket,"mnistdocker/x_train","/tmp/x_train.npy")
    x_train = np.load("/tmp/x_train.npy")
    
    minio_client.fget_object(minio_bucket,"mnistdocker/y_train","/tmp/y_train.npy")
    y_train = np.load("/tmp/y_train.npy")
    
    #fit the model and return the history while training
    history = model.fit(
      x=x_train,
      y=y_train,
      epochs=no_epochs,
      batch_size=20,
    )
    
    minio_client.fget_object(minio_bucket,"mnistdocker/x_test","/tmp/x_test.npy")
    x_test = np.load("/tmp/x_test.npy")
    
    minio_client.fget_object(minio_bucket,"mnistdocker/y_test","/tmp/y_test.npy")
    y_test = np.load("/tmp/y_test.npy")
    

    # Test the model against the test dataset
    # Returns the loss value & metrics values for the model in test mode.
    model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
    
    # Confusion Matrix

    # Generates output predictions for the input samples.
    test_predictions = model.predict(x=x_test)

    # Returns the indices of the maximum values along an axis.
    test_predictions = np.argmax(test_predictions,axis=1) # the prediction outputs 10 values, we take the index number of the highest value, which is the prediction of the model

    # generate confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions=test_predictions)
    confusion_matrix = confusion_matrix.numpy()
    vocab = list(np.unique(y_test))
    data = []
    for target_index, target_row in enumerate(confusion_matrix):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_csv = df_cm.to_csv(header=False, index=False)
    
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                  ],
                "target_col" : "actual",
                "predicted_col" : "predicted",
                "source": cm_csv,
                "storage": "inline",
                "labels": [0,1,2,3,4,5,6,7,8,9]
            },
            {
                'storage': 'inline',
                'source': '''# Model Overview
## Model Summary

```
{}
```

## Model Performance

**Accuracy**: {}
**Loss**: {}

'''.format(metric_model_summary,model_accuracy,model_loss),
                'type': 'markdown',
            }
        ]
    }
    
    metrics = {
      'metrics': [{
          'name': 'model_accuracy',
          'numberValue':  float(model_accuracy),
          'format' : "PERCENTAGE"
        },{
          'name': 'model_loss',
          'numberValue':  float(model_loss),
          'format' : "PERCENTAGE"
        }]}
    
    ### Save model to minIO
    
    keras.models.save_model(model,"/tmp/detect-digits")

    minio_client = Minio(
            "10.0.102.158:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
    minio_bucket = "mlpipeline"

    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                minio_client.fput_object(bucket_name, remote_path, local_file)

    upload_local_directory_to_minio("/tmp/detect-digits",minio_bucket,"mnistdocker/models/detect-digits/1/") # 1 for version 1
    
    print("Saved model to minIO")
    
    output = namedtuple('output', ['mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    return output(json.dumps(metadata),json.dumps(metrics))

if __name__ == '__main__':
    model_building()