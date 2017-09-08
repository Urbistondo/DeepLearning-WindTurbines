import pandas as pd
from util import plot_handler as ph


data = pd.read_csv('loss.csv')
ph.show_plot(data.iloc[:,0], data.iloc[:,1], 'Loss')