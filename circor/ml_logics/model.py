from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.metrics import Recall
import numpy as np
from typing import Tuple


def init_model(X: np.ndarray) -> Model:
    base_model = DenseNet201(include_top=False, input_shape=X.shape[1:])

    base_model.trainable = False

    layer1=Flatten()
    layer2=Dense(2048, activation='relu')
    layer3=Dense(1024, activation='relu')
    layer4=Dense(512, activation='relu')
    layer5=Dense(256, activation='relu')
    layer6=Dense(128, activation='relu')
    layer7=Dense(64, activation='relu')
    layer8=Dense(32, activation='relu')
    layer9=Dense(16, activation='relu')
    layer10=Dense(8, activation='relu')
    layer11=Dense(1, activation='sigmoid')

    model = Sequential([base_model, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11])

    print("\n✅ model initialized")
    return model

def compile_model(model: Model, learning_rate=0.0001) -> Model:
    """
    Compile the Neural Network
    """
    adam_opt = Adam(learning_rate=learning_rate)
    model.compile(loss=BinaryCrossentropy(),
                  optimizer=adam_opt,
                  metrics=['accuracy',Recall()])

    print("\n✅ model compiled")
    return model

def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=32,
                patience=10,
                validation_split=0.2,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    es = EarlyStopping(monitor="val_recall",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history

def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=32) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """
    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy=metrics["accuracy"]
    recall = metrics["recall"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)}, accuracy {round(accuracy, 2)}, recall {round(recall, 2)}")

    return metrics
