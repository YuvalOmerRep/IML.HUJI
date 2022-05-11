import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from utils import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import floor
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)

    losses = [model.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]

    px.line(x=[i for i in range(1, n_learners + 1)],
            y=losses, title="Adaboost test error for ensemble size").add_trace(
        go.Scatter(x=[i for i in range(1, n_learners + 1)],
                   y=[model.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)],
                   mode="lines", name="train loss")).update_layout(xaxis_title="ensemble size",
                                                                   yaxis_title="error").write_image(
        f"./loss_per_iterations_with_{noise}_noise.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(2, 2, subplot_titles=["decision surface for ensemble of size 5",
                                              "decision surface for ensemble of size 50",
                                              "decision surface for ensemble of size 100",
                                              "decision surface for ensemble of size 250"])

    for index, i in enumerate(T):
        fig.add_traces([go.Scatter(mode="markers", x=test_X[:, 0], y=test_X[:, 1],
                                   marker=dict(color=(test_y == 1).astype(int), colorscale=[custom[0], custom[-1]])),
                        decision_surface(lambda x: model.partial_predict(x, i), lims[0], lims[1],
                                         showscale=False)],
                       rows=floor(index / 2) + 1, cols=index % 2 + 1)

    fig.write_image(f"./decision_surface_with_{noise}_noise.png")

    # Question 3: Decision surface of best performing ensemble
    best = np.argmin(losses)

    fig1 = go.Figure()
    fig1.add_traces([go.Scatter(mode="markers", x=test_X[:, 0], y=test_X[:, 1],
                                marker=dict(color=(test_y == 1).astype(int), colorscale=[custom[0], custom[-1]])),
                     decision_surface(lambda x: model.partial_predict(x, best + 1), lims[0], lims[1],
                                      showscale=False)]).update_layout(
        title=f"Best ensemble of size {best + 1}"
              f" with accuracy {accuracy(test_y, model.partial_predict(test_X, best + 1))}")

    fig1.write_image(f"./decision_surface_best_with_{noise}_noise.png")

    # Question 4: Decision surface with weighted samples
    fig2 = px.scatter(x=train_X[:, 0], y=train_X[:, 1], size=model.D_).update_traces(
        marker=dict(color=(train_y == 1).astype(int), colorscale=[custom[0], custom[-1]])).add_trace(
        decision_surface(lambda x: model.predict(x), lims[0], lims[1], showscale=False))

    fig2.update_layout(title=f"Full ensemble decision surface with weights on markers")

    fig2.write_image(f"./decision_surface_all_sizes_{noise}_noise.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)
