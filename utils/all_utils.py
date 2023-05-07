import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
plt.style.use("fivethirtyeight")


def prepare_data(df, target_col="y"):
    """
    The function prepares data by separating the target column from the rest of the dataframe.
    
    :param df: a pandas DataFrame containing the data to be prepared
    :param target_col: The name of the target column in the dataframe. This column contains the values
    that we want to predict or classify. In this function, it is set to "y" by default, but it can be
    changed to any other column name if needed, defaults to y (optional)
    :return: The function `prepare_data` returns two variables, `X` and `y`. `X` is a pandas DataFrame
    that contains all the columns of the input DataFrame `df` except for the `target_col` column. `y` is
    a pandas Series that contains the values of the `target_col` column of the input DataFrame `df`.
    """
    X = df.drop(target_col, axis=1)

    y = df[target_col]

    return X, y



def save_plots(df, model, filename="plot.png", plot_dir="plots"):
    """
    This function saves a scatter plot with decision regions based on a trained machine learning model
    and input data.
    
    :param df: The input dataframe containing the data to be plotted
    :param model: The machine learning model that has been trained on the data and will be used to make
    predictions on new data
    :param filename: The name of the file to save the plot as. If not specified, it will default to
    "plot.png", defaults to plot.png (optional)
    :param plot_dir: The directory where the plot will be saved. If the directory does not exist, it
    will be created, defaults to plots (optional)
    """
    def _create_base_plot(df):
        """
        This function creates a scatter plot with x1 and x2 as the x and y axes, respectively, and
        colors the points based on the y variable, while also adding horizontal and vertical lines at 0.
        
        :param df: The input dataframe containing the data to be plotted
        """
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        
        figure = plt.gcf()
        figure.set_size_inches(10, 8)
    
    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        """
        This function plots the decision regions of a classifier based on the input data and resolution.
        
        :param X: The input features as a pandas DataFrame
        :param y: The target variable or the dependent variable of the dataset
        :param classifier: The machine learning classifier that has been trained on the data and will be
        used to make predictions on new data
        :param resolution: The resolution parameter determines the step size of the meshgrid used to
        create the decision boundary plot. A smaller value will result in a more detailed plot, but may
        also increase the computation time
        """
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)
        
        X = X.values # as an array
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        x1_min, x1_max = x1.min() - 1, x1.max() + 1 
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                              )
        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.plot()
        
    X, y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)