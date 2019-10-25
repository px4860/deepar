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
data['promotion'] = 0
data.loc[data['order'] > 150, 'promotion'] = 1
order_max = data['order'].max()
order_min = data['order'].min()
data[['order','count']] = data[['order','count']].apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.drop(columns=['seller','marketplace'], inplace=True)

ts = TimeSeries(data.head(500))

# ts = MockTs()
dp_model = DeepAR(ts, epochs=100)
dp_model.init()
dp_model.model.load_weights('1.h5', by_name=True)
# dp_model.more_fit()


def sigmoid(x):
    y = 1/(1+np.exp(-1*x))
    return y


def tanh(x):
    y = 2*sigmoid(2*x)-1
    # y = (np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))
    return y


predict_data = data.loc['2019-06-01':'2019-06-30']
true_data = data.loc['2019-07-01':'2019-07-30']


def decode(predict_data, true_data):
    # input_shape = 1,30,3
    # LSTM output = 4 dim
    tot_res = [predict_data.iloc[-1]]
    a = K.function(inputs=[dp_model.model.input], outputs=dp_model.model.get_layer('LSTM').output)
    H, ht_1, C_t_1 = a([np.array(predict_data).reshape(1, 30, 3)])
    weights = dp_model.model.get_layer('LSTM').get_weights()
    weights_G = dp_model.model.get_layer('main_output').get_weights()

    bias_i = weights[2][:4]
    bias_f = weights[2][4:8]
    bias_C = weights[2][8:12]
    bias_o = weights[2][12:]
    for i in range(true_data.shape[0]):
        input_xt = np.dot(tot_res[-1], weights[0])
        input_ht_1 = np.dot(ht_1, weights[1])
        input = input_xt+input_ht_1

        it = sigmoid(input[:, :4] + bias_i)
        ft = sigmoid(input[:, 4:8]+bias_f)
        Ct = tanh(input[:, 8:12]+bias_C)
        Ct = ft*C_t_1+it*Ct
        ot = sigmoid(input[:, 12:]+bias_o)
        ht = ot*tanh(Ct)
        mu = np.dot(ht, weights_G[0]) + weights_G[2]
        sigma = np.log(1 + np.exp(np.dot(ht, weights_G[1]) + weights_G[3]))
        tot_res.append(pd.Series([normal(loc=mu, scale=np.sqrt(sigma), size=1)[0], true_data.iloc[i]['count'],
                                  true_data.iloc[i]['promotion']], index=['order', 'count', 'promotion']))

        ht_1 = ht
        C_t_1 = Ct
    return tot_res

# def get_sample_prediction(sample, fn):
#     sample = np.array(sample).reshape(1, 30, 3)
#     output = fn([sample])
#     samples = []
#     # return output[0].reshape(1)
#     for mu,sigma in zip(output[0].reshape(30), output[1].reshape(30)):
#         samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
#     return np.array(samples)
#
# def forcast(input_data):
#     ress = []
#     for i in range(301):
#         ress.append(get_sample_prediction(input_data, dp_model.predict_theta_from_input))
#     return ress
#
#
# for i in tqdm.tqdm(range(30)):
#     if not tot_res:
#         tot_res.append(pd.DataFrame(forcast(predict_data)).T.tail(1))
#     else:
#         tot_res.append(pd.DataFrame(forcast(np.concatenate([np.array(predict_data.tail(30 - i)), np.c_[np.array([np.percentile(x, 90) for x in tot_res]), np.array(true_data.loc[:,'count':'promotion'].head(i))]]))).T.tail(1))

# res_df = pd.DataFrame()
# for i in range(100):
#     res_df = pd.concat([res_df, pd.DataFrame(forcast(predict_data)).T])
# tot_res = res_df.T.tail(7).reset_index(drop=True)

# plt.plot(batch[1].reshape(1), linewidth=6)
tot_res = []
for i in range(21):
    ress = decode(predict_data,true_data)
    tot_res.append(pd.DataFrame(ress)['order'].reset_index(drop=True).tail(30))
# tot_res = pd.concat(tot_res).reset_index(drop=True)
tot_res = pd.DataFrame(tot_res).reset_index(drop=True).T.reset_index(drop=True)
a = tot_res.apply(np.sum)
tot_res['predict'] = tot_res.loc[:,a[a == np.percentile(tot_res.apply(np.sum), 50)].index]
tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
#
# plt.plot(res_df.T.tail(7).reset_index(drop=True), label='predict')
tot_res = tot_res.apply(lambda x: x*(order_max-order_min)+order_min)
true_data = true_data.apply(lambda x: x*(order_max-order_min)+order_min)
plt.plot(true_data.reset_index(drop=True)['order'], label='true')
plt.plot(tot_res.predict, label='predict')
plt.plot(tot_res.mu, linewidth=2, label='mu')
plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
plt.title('Prediction uncertainty %s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER.csv')

plt.legend()
plt.show()

print((np.percentile(tot_res.apply(np.sum), 90)-true_data.apply(np.sum))/true_data.apply(np.sum))