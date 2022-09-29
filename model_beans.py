from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam

from model_eval import model_eval
from model_eval.model_eval import plot_model_hist
from preprocessing import preprocessing
from utils import utils

# %%

config = utils.get_config()
# constroi o dataset
valid_data = preprocessing.load_tf_dataset("Beans", "validation")
valid_data = preprocessing.prepair_ds(
    valid_data,
)

test_data = preprocessing.load_tf_dataset("Beans", "test")
test_data = preprocessing.prepair_ds(tensor=test_data)

train_ = preprocessing.load_tf_dataset("Beans", "train")
dataset = preprocessing.prepair_ds(tensor=train_)
# dataset = dataset.map(preprocessing.generate_data_train)

STEPS = len(train_) / config["BATCH_SIZE"]

layers = [
    Input(config["TENSOR_SPEC"], name="image"),
    Conv2D(filters=64, kernel_size=3, activation="relu"),
    MaxPool2D(),
    Conv2D(64, kernel_size=3, activation="relu"),
    MaxPool2D(),
    Conv2D(filters=128, kernel_size=3, activation="relu"),
    MaxPool2D(),
    Conv2D(128, kernel_size=3, activation="relu"),
    MaxPool2D(),
    Conv2D(128, kernel_size=3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    # Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax"),
]

modelo = Sequential(layers=layers, name="CONV_net")
modelo.compile(
    loss=CategoricalCrossentropy(),
    metrics=CategoricalAccuracy(),
    optimizer=Adam(),
)

print(model_eval.model_info(model=modelo))


history = modelo.fit(
    dataset,
    steps_per_epoch=STEPS,
    validation_data=valid_data,
    validation_steps=STEPS,
    epochs=config["EPOCHS"],
    callbacks=EarlyStopping(monitor="categorical_accuracy", patience=1),
)
plot_model_hist(history=history)

y_pred = modelo.predict(test_data, steps=STEPS)
y_true = preprocessing.label_to_nparray(tensor=test_data, steps=STEPS)
model_eval.model_out(y_pred=y_pred, y_true=y_true)
