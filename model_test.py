from deepar.dataset.time_series import MockTs,TimeSeries
from deepar.model.lstm import DeepAR
from keras.models import load_model

from numpy.random import normal
import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from deepar.model.layers import GaussianLayer
from deepar.model.loss import gaussian_likelihood,gaussian_likelihood_2
import os


data = pd.read_csv(os.getcwd()+'\data\%s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER-1.csv', header=None, names=['date','order','seller','marketplace'])
data['count'] = data['date'].apply(lambda x: int(x.split('-')[0])*10000+int(x.split('-')[1])*100+int(x.split('-')[2]))
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.drop(columns=['seller','marketplace'], inplace=True)
data['promotion'] = 0
data.loc[data['order'] > 150, 'promotion'] = 1

ts = TimeSeries(data.head(500))

# ts = MockTs()
dp_model = DeepAR(ts, epochs=100)
dp_model.init()
dp_model.model.load_weights('1.h5', by_name=True)
# dp_model.more_fit()


def get_sample_prediction(sample, fn):
    sample = np.array(sample).reshape(1, 30, 3)
    output = fn([sample])
    samples = []
    # return output[0].reshape(1)
    for mu,sigma in zip(output[0].reshape(30), output[1].reshape(30)):
        samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
    return np.array(samples)


predict_data = data.loc['2019-07-05':'2019-08-03']
true_data = data.loc['2019-08-04':'2019-09-02']
tot_res =[]


def forcast(input_data):
    ress = []
    for i in range(301):
        ress.append(get_sample_prediction(input_data, dp_model.predict_theta_from_input))
    return ress


for i in tqdm.tqdm(range(30)):
    if not tot_res:
        tot_res.append(pd.DataFrame(forcast(predict_data)).T.tail(1))
    else:
        tot_res.append(pd.DataFrame(forcast(np.concatenate([np.array(predict_data.tail(30 - i)), np.c_[np.array([np.percentile(x, 90) for x in tot_res]), np.array(true_data.loc[:,'count':'promotion'].head(i))]]))).T.tail(1))

# res_df = pd.DataFrame()
# for i in range(100):
#     res_df = pd.concat([res_df, pd.DataFrame(forcast(predict_data)).T])
# tot_res = res_df.T.tail(7).reset_index(drop=True)

# plt.plot(batch[1].reshape(1), linewidth=6)
tot_res = pd.concat(tot_res).reset_index(drop=True)
tot_res['predict'] = tot_res[np.where(tot_res.apply(np.sum)==np.percentile(tot_res.apply(np.sum), 90))[0][0]]
tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
#
# plt.plot(res_df.T.tail(7).reset_index(drop=True), label='predict')
plt.plot(true_data.reset_index(drop=True)['order'], label='true')
plt.plot(tot_res.predict, label='predict')
plt.plot(tot_res.mu, linewidth=2, label='mu')
plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
plt.title('Prediction uncertainty %s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER.csv')

plt.legend()
plt.show()

print((np.percentile(tot_res.apply(np.sum), 90)-true_data.apply(np.sum))/true_data.apply(np.sum))