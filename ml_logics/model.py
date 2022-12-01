from tensorflow.keras import Sequential, layers,optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall
import numpy as np
from typing import Tuple


def init_model():
    model = Sequential()
    model.add(layers.Masking(mask_value=-10, input_shape=(1025,150,4)))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    print("\n✅ model initialized")
    return model

def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Neural Network
    """
    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=adam_opt,metrics=['accuracy',Recall()])
    print("\n✅ model compiled")
    return model

def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=32,
                patience=5,
                validation_split=0.2,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history

def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
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
    recall = metrics["recall_1"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)}, accuracy {round(accuracy, 2)}, recall {round(recall, 2)}")

    return metrics
