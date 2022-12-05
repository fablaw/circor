from circor.preprocessing.preprocessing_csv import select_patients
from circor.preprocessing.preprocess_denoise import download_reconstruct_upload, synchronise, reconstruct_signal, neigh_block, nb_beta
from circor.preprocessing.preprocess_rnn import wav_to_1D
from circor.ml_logics.data import rgba_data, rgba_new
from circor.ml_logics.model import init_model, compile_model, train_model, evaluate_model
from sklearn.model_selection import train_test_split
import glob
import time
import numpy as np
import os
from google.cloud import storage

project=os.environ.get("PROJECT")
bucket_name=os.environ.get("BUCKET_NAME")
storage_client = storage.Client(project=project)
bucket = storage_client.get_bucket(bucket_name)

def preprocess():

    ''' applying denoising functions on the original wave form, and output cleaned waveform  '''
    print("\n⭐️ Use case: preprocess")

    tsv_blob = bucket.blob(f"tsv_raw")
    csv_blob = bucket.blob(f"training_data.csv")
    wav_blob = bucket.blob(f"audio_raw")

    wave_npy=wav_to_1D(wav_blob)

    df_final=select_patients(tsv_blob)











    download_reconstruct_upload

    return None


def train():
    """train a model on preprocessed data"""
    print("\n⭐️ Use case: train")

    X=rgba_data[0]
    y=rgba_data[1]

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1)

    #model params
    learning_rate=0.0001
    batch_size=32
    patience=10

    model=init_model(X_train)
    model=compile_model(model, learning_rate=learning_rate)
    model, history=train_model(model,
                               X_train,
                               y_train,
                               batch_size=batch_size,
                               patience=patience,
                               validation_split=0.2,
                               )

    val_acc=np.max(history.history['val_accuracy'])
    print(f"Validation set accuracy: {round(val_acc,2)}")
    val_recall=np.max(history.history['val_recall'])
    print(f"Validation set recall: {round(val_recall,2)}")

    return (model, X_test, y_test)

def evaluate():
    '''evaluate on testing set'''
    print("\n⭐️ Use case: evaluate")

    metrics=evaluate_model(train()[0], #-> model
                           train()[1],   #-> X_test
                           train()[2],   #-> y_test
                           batch_size=32
                          )

    print(f"Testing set accuracy: {round(metrics['accuracy'],2)}")
    print(f"Testing set recall: {round(metrics['recall'],2)}")

    return None

def pred():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: predict")

    X_processed=rgba_new()
    model=train()[0]
    y_pred=model.predict(X_processed)

    print("\n✅ prediction done!")

    if round(y_pred,2) >= 0.50:
        return f"There is a high probability that murmurs exist"
    return f"There is a high probability that murmurs do not exist"

if __name__ == '__main__':

    preprocess
    train
    evaluate
    pred
