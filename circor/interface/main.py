
from circor.preprocessing.preprocess_sync import download_reconstruct_upload
from circor.preprocessing.main_1D import download_to_local
from circor.ml_logics.data import rgba_data, rgba_new
from circor.ml_logics.model import init_model, compile_model, train_model, evaluate_model
from sklearn.model_selection import train_test_split
from circor.ml_logics.registry import save_model, load_model, get_model_version
import numpy as np
import time
import os

timestamp = time.strftime('%d_%H_%M')

def preprocess():

    ''' applying denoising functions on the original wave form, and output cleaned waveform  '''
    print("\n⭐️ Use case: preprocess")

    download_to_local()

    download_reconstruct_upload()

    return None

def train():
    """train a model on preprocessed data"""
    print("\n⭐️ Use case: train")
    X, y=rgba_data()

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1)

    output_X=os.getcwd()+'/processed_data/X_test/X_test.npy'
    np.save(output_X, X_test)

    output_y=os.getcwd()+'/processed_data/y_test/y_test.npy'
    np.save(output_y, y_test)

    print("\n⭐️ Testing set saved")

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

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",

        # Data source
        model_version=get_model_version(),
        dataset_timestamp=timestamp
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(acc=val_acc))

    return val_acc

def evaluate():
    '''evaluate on testing set'''
    print("\n⭐️ Use case: evaluate")

    X_file=os.getcwd()+'/processed_data/X_test/X_test.npy'
    y_file=os.getcwd()+'/processed_data/y_test/y_test.npy'

    X_test=np.load(X_file)
    y_test=np.load(y_file)

    model=load_model()
    metrics=evaluate_model(model,
                           X_test,
                           y_test
                           )

    print(f"Testing set accuracy: {round(metrics['accuracy'],2)}")
    print(f"Testing set recall: {round(metrics['recall'],2)}")

    acc=round(metrics['recall'],2)

    # Save evaluation
    params = dict(
        dataset_timestamp=timestamp,
        model_version=get_model_version(),

        # Package behavior
        context="evaluate"
        )

    save_model(params=params, metrics=dict(acc=acc))

    return acc

def pred(X_pred=None, model=None):
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: predict")

    if X_pred==None:
        X_file=os.getcwd()+'/processed_data/X_test/X_test.npy'
        X_test=np.load(X_file)

        X_processed=X_test[0,:,:,:]
        X_goodshape=np.expand_dims(X_processed, axis=0)
    else:
        X_goodshape=rgba_new(X_pred)

    y_pred=model.predict(X_goodshape)

    print("\n✅ prediction done!")

    if round(y_pred[0][0],2) >= 0.50:
        print("Murmurs exist!")
    else:
        print("There is a high probability that murmurs do not exist.")

    return None


if __name__ == '__main__':

    preprocess()
    train()
    evaluate()
    pred()
