
from circor.preprocessing.preprocess_sync import download_reconstruct_upload
from circor.ml_logics.data import rgba_new
from circor.ml_logics.model import init_model, compile_model, train_model, evaluate_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import numpy as np


def preprocess():

    ''' applying denoising functions on the original wave form, and output cleaned waveform  '''
    print("\n⭐️ Use case: preprocess")

    X, y = download_reconstruct_upload()

    return X, y

def train():
    """train a model on preprocessed data"""

    X, y=preprocess()

    print("\n⭐️ Use case: train")
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1)

    output_X=f'circor/processed_data/X_test.npy'
    np.save(output_X, X_test)

    output_y=f'circor/processed_data/y_test.npy'
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

    # Save model
    model.save('circor/saved_model')
    print("\n✅ model saved!")

    return val_acc

def evaluate():
    '''evaluate on testing set'''
    print("\n⭐️ Use case: evaluate")

    X_file=f'circor/processed_data/X_test.npy'
    y_file=f'circor/processed_data/y_test.npy'

    X_test=np.load(X_file)
    y_test=np.load(y_file)

    model=models.load_model('circor/saved_model')
    print("\n✅ model loaded!")

    metrics=evaluate_model(model,
                           X_test,
                           y_test
                           )

    print(f"Testing set accuracy: {round(metrics['accuracy'],2)}")
    print(f"Testing set recall: {round(metrics['recall'],2)}")

    acc=round(metrics['recall'],2)

    return acc

def pred(X_pred=None, model=None):
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Use case: predict")

    if X_pred==None:
        X_file=f'circor/processed_data/X_test.npy'
        X_test=np.load(X_file)

        X_processed=X_test[0,:,:,:]
        X_goodshape=np.expand_dims(X_processed, axis=0)
    else:
        X_goodshape=rgba_new(X_pred)

    model=models.load_model('circor/saved_model')
    print("\n✅ model loaded!")

    y_pred=model.predict(X_goodshape)

    res=np.round(y_pred[0][0],2)

    print(f"\n✅ prediction done: {res} ")

    if res >= 0.50:
        return f"Murmurs detected!, chance : {np.round(res*100, 4)}%"
    else:
        return f"Murmurs not detected, chance : {np.round((1-res)*100,4)}%"



if __name__ == '__main__':

    train()
    evaluate()
    pred()
