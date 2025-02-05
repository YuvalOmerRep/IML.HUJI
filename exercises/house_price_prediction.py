from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    full_data = pd.read_csv(filename).dropna().drop_duplicates()

    full_data.drop(full_data[full_data.price < 1].index, inplace=True)

    labels = full_data["price"]

    # features = full_data[["bedrooms",
    #                       "bathrooms",
    #                       "sqft_living",
    #                       "sqft_lot",
    #                       "floors",
    #                       "waterfront",
    #                       "view",
    #                       "condition",
    #                       "grade",
    #                       "sqft_above",
    #                       "sqft_basement",
    #                       "yr_built",
    #                       "yr_renovated",
    #                       "lat",
    #                       "zipcode",
    #                       "long",
    #                       "sqft_living15",
    #                       "sqft_lot15"]]

    features = full_data[["bedrooms",
                          "bathrooms",
                          "sqft_living",
                          "floors",
                          "yr_built",
                          "yr_renovated"]]

    features["yr_renovated"] = features["yr_renovated"].mask(features["yr_renovated"] <= 0,
                                                             features["yr_built"], axis=0)

    # print(features["yr_renovated"])

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_correlation_values = np.apply_along_axis(pearson_correlation, 0, X, np.array(y))

    for index, feature in enumerate(X.keys()):
        go.Figure().add_trace(
            go.Scatter(x=X[feature], y=y,
                       mode="markers")).update_layout(title={
            "text": f"{feature} with Pearson Correlation "
                    f"{pearson_correlation_values[index]}"}).write_image(f"{output_path}/{feature}.png")


def pearson_correlation(X: np.array, y: np.array) -> float:
    return (np.cov(X, y) / (np.std(X) * np.std(y)))[0][1]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    house_features, house_prices = load_data(
        "../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(house_features, house_prices)

    regressor = LinearRegression()

    regressor.fit(house_features, house_prices)

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(house_features, house_prices)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    regressor = LinearRegression()
    #
    # for i in range(10, 101):
    #     regressor.fit(x_train.sample(frac=))
