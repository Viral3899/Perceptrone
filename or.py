from utils.all_utils import save_plots, prepare_data

from utils.model import Perceptron

import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
import sys
import os

gate = "OR gate"
logging.info(f"started logging for {gate}")

def main(data, modelName, plotName, eta, epochs):
    """
    This function prepares data, fits a Perceptron model, saves the model, and generates plots.
    
    :param data: The input data for the model, in the form of a pandas DataFrame
    :param modelName: The name of the file to save the trained model as
    :param plotName: The plotName parameter is a string that represents the name of the plot that will
    be saved after running the save_plots function
    :param eta: eta is the learning rate of the Perceptron algorithm. It determines the step size at
    each iteration while moving toward a minimum of a loss function. A higher learning rate can result
    in faster convergence, but it can also cause the algorithm to overshoot the minimum and fail to
    converge
    :param epochs: The number of times the entire dataset is passed through the model during training
    """
    df = pd.DataFrame(data)
    logging.info(f'This is the raw Data - >\n{df}')
    X, y = prepare_data(df)
   
    model = Perceptron(epochs=epochs, eta=eta)

    model.fit(X=X, y=y)

    _ = model.total_loss()
    
    model.saving_model(file_name=modelName, model_dir='models')

    save_plots(df, model, plotName)


if __name__ == "__main__":
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y":  [0, 1, 1, 1]
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(f'{">"*10}started Training for {gate} {"<"*10}')
        main(data=OR,plotName='or.png',modelName='or.model',eta=ETA,epochs=EPOCHS)
        logging.info(f'{"<"*10} Done Training for {gate} {">"*10}')
         
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)        