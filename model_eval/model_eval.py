from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import f1_score


def model_info(model):
    return model.summary()


def history_2df(history, name="history.csv"):
    result = pd.DataFrame(history.history)
    result.to_csv(name, index=False)
    return result


def plot_model_hist(history):
    result = history_2df(history=history)
    sns.lineplot(result)
    plt.show()


def model_out(y_pred, y_true, name="pred_test.csv"):
    assert (
        y_pred.shape[0] == y_true.shape[0]
    ), f"[DIM ERROR] y_pred({y_pred.shape[0]}) != y_true({y_true.shape[0]}): dimensoes diferentes"
    result = pd.DataFrame()
    result["y_pred"] = np.argmax(y_pred, axis=-1)
    result["y_true"] = np.argmax(y_true, axis=-1)
    result.to_csv(name, index=False)
