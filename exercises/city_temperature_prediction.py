from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).drop_duplicates()

    full_data = full_data.drop(full_data[full_data["Temp"] < -40].index)

    full_data["DayOfYear"] = full_data["Date"].apply(lambda x: x.timetuple().tm_yday - 1)

    data = full_data[["Country",
                      "City",
                      "Date",
                      "Year",
                      "Month",
                      "Day",
                      "DayOfYear",
                      "Temp"]]

    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[["Temp", "DayOfYear", "Year", "Month"]][data["Country"] == "Israel"]
    israel_data["Year"] = israel_data["Year"].astype(str)

    px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
               title="Temp by Day of year between the years 1995-2007").write_image("./graphs/DaysToTemp.png")

    px.bar(israel_data[["Month", "Temp"]].groupby("Month").agg("std"), text_auto=True,
           title="standard deviation of Temp across months in israel").write_image("./graphs/MonthStd.png")

    # Question 3 - Exploring differences between countries

    data_group = data.groupby(["Country", "Month"])[["Temp"]].agg(["mean", "std"])

    data_group.columns = ["Temp_mean", "Temp_std"]
    data_group = data_group.reset_index()

    px.line(data_group, x="Month", y="Temp_mean", error_y="Temp_std", color="Country",
            title="Mean Temp by month in each country and deviation of each month").write_image("./graphs/std.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_data["DayOfYear"], israel_data["Temp"])

    loss = []
    for i in range(1, 11):
        model = PolynomialFitting(i)
        model.fit(np.array(train_X), np.array(train_y))
        loss.append(round(model.loss(np.array(test_X), np.array(test_y)), 2))

    print(loss)
    px.bar(x=[i for i in range(1, 11)],
           y=loss, text_auto=True).update_layout(title="Loss by polynomial fitting degree",
                                                 xaxis_title="Degree of polynomial", yaxis_title="Loss").write_image(
        "./graphs/ks.png")

    # Question 5 - Evaluating fitted model on different countries
    nether_data = data[["Temp", "DayOfYear"]][data["Country"] == "The Netherlands"]
    jordan_data = data[["Temp", "DayOfYear"]][data["Country"] == "Jordan"]
    south_data = data[["Temp", "DayOfYear"]][data["Country"] == "South Africa"]

    model = PolynomialFitting(6)
    model.fit(israel_data["DayOfYear"], israel_data["Temp"])

    px.bar(x=["Netherlands", "Jordan", "South Africa"],
           y=[model.loss(nether_data["DayOfYear"], nether_data["Temp"]),
              model.loss(jordan_data["DayOfYear"], jordan_data["Temp"]),
              model.loss(south_data["DayOfYear"], south_data["Temp"])],
           color={"Netherlands": "Netherlands",
                  "Jordan": "Jordan",
                  "South Africa": "South Africa"}, text_auto=True).update_layout(
        title="Loss of fitted model over different countries", xaxis_title="Country", yaxis_title="Loss"
    ).write_image("./graphs/loss_fitted.png")
