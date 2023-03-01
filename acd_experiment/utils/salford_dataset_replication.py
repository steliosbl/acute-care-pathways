import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(
    y_true,
    y_pred,
    ax,
    display_labels=[1, 0],
    xlabel="True Class",
    ylabel="Predicted Class",
    plot_title="Confusion Matrix",
    normalize="true",
):
    ax.grid(False)
    values_format = ".2%" if normalize else None
    cm_fig = ConfusionMatrixDisplay(
        np.rot90(np.flipud(confusion_matrix(y_true, y_pred, normalize=normalize))),
        display_labels=display_labels,
    ).plot(
        values_format=values_format,
        ax=ax,
        # cmap="Purples"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
