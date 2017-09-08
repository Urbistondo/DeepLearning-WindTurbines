import pandas as pd
from util import plot_handler as ph


data = pd.read_csv('../output/top/normal/WGENBearDETemp/9/27037/WGENBearDETemp/predictions.csv')
ph.show_plot(data.iloc[:,1], data.iloc[:,2], 'WGENBearDETemp')

data = pd.read_csv('../output/top/normal/WGENBearDETemp/9/27037/WGENBearNDETemp/predictions.csv')
ph.show_plot(data.iloc[:,1], data.iloc[:,2], 'WGENBearNDETemp')