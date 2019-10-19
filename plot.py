from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

data = pd.read_csv(os.getcwd()+'\data\%s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER-1.csv', header=None, names=['date','order','seller','marketplace'])
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.drop(columns=['seller', 'marketplace'], inplace=True)
# data['promotion'] = 0
# data.loc[data['order'] > 220, 'promotion'] = 1
# data = data.reset_index(drop=True)
plt.plot(data)
plt.show()