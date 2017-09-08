import pandas as pd
from util import plot_handler as ph


data = pd.read_csv('loss.csv')
ph.show_plot(data.iloc[:,1], data.iloc[:,2], 'Loss')