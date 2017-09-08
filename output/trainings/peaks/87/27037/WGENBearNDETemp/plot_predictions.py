import pandas as pd
from util import plot_handler as ph


data = pd.read_csv('predictions.csv')
ph.show_plot(data.iloc[:,1], data.iloc[:,2], 'WGENBearNDETemp', 'Model predictions')