import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def plot_decision_boundry(model_to_plot: nn.Module, x_values: torch.Tensor, y_values: torch.Tensor):
    model_to_plot.to("cpu")
    x_values, y_values = x_values.to("cpu"), y_values.to("cpu")

    x_min, x_max, = x_values[:, 0].min() - 0.1, x_values[:, 0].max() + 0.1
    y_min, y_max, = x_values[:, 1].min() - 0.1, x_values[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    x_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model_to_plot.eval()
    with torch.inference_mode():
        y_logits_ = model_to_plot(x_to_pred_on)

    if len(torch.unique(y_logits_)) > 2:
        y_pred = torch.softmax(y_logits_, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits_))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(x_values[:, 0], y_values[:, 1], c=y_values, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
