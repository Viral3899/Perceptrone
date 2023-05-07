from utils.all_utils import save_plots, prepare_data

from utils.model import Perceptron

import pandas as pd
import numpy as np


def main(data, modelName, plotName, eta, epochs):
    df_OR = pd.DataFrame(data)
    # print(df_OR)
    X, y = prepare_data(df_OR)
   
    model_OR = Perceptron(epochs=epochs, eta=eta)

    model_OR.fit(X=X, y=y)

    _ = model_OR.total_loss()

    model_OR.saving_model(file_name=modelName, model_dir='models')

    save_plots(df_OR, model_OR, plotName)


if __name__ == "__main__":
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y":  [0, 1, 1, 1]
    }
    ETA = 0.3
    EPOCHS = 10
    
    main(data=OR,plotName='or.png',modelName='or.model',eta=ETA,epochs=EPOCHS)