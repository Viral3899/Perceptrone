from utils.all_utils import save_plots, prepare_data

from utils.model import Perceptron

import pandas as pd
import numpy as np


def main(data, modelName, plotName, eta, epochs):
    df_and = pd.DataFrame(data)
    # print(df_and)
    X, y = prepare_data(df_and)
   
    model_and = Perceptron(epochs=epochs, eta=eta)

    model_and.fit(X=X, y=y)

    _ = model_and.total_loss()

    model_and.saving_model(file_name=modelName, model_dir='models')

    save_plots(df_and, model_and, plotName)


if __name__ == "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y":  [0, 0, 0, 1]
    }
    ETA = 0.2
    EPOCHS = 15
    
    main(data=AND,plotName='and.png',modelName='and.model',eta=ETA,epochs=EPOCHS)