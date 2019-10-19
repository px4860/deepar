# %load_ext autoreload
# %autoreload 2

from deepar.dataset.time_series import MockTs,TimeSeries
from deepar.model.lstm import DeepAR

from numpy.random import normal
import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


data = pd.read_csv(os.getcwd()+'\data\%s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER-1.csv', header=None, names=['date','order','seller','marketplace'])
data['count'] = data['date'].apply(lambda x: int(x.split('-')[0])*10000+int(x.split('-')[1])*100+int(x.split('-')[2]))
data[['count']].apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.drop(columns=['seller','marketplace'], inplace=True)
data['promotion'] = 0
data.loc[data['order'] > 150, 'promotion'] = 1

ts = TimeSeries(data.head(500))

# ts = MockTs()
dp_model = DeepAR(ts, epochs=150)
dp_model.instantiate_and_fit()


def get_sample_prediction(sample, fn):
    sample = np.array(sample).reshape(1, 30, 3)
    output = fn([sample])
    samples = []
    # return output[0].reshape(1)
    for mu,sigma in zip(output[0].reshape(30), output[1].reshape(30)):
        samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
    return np.array(samples)


# predict_data = ts.next_batch(1, 50)[0]
predict_data = data.loc['2019-05-07':'2019-06-05']
# predict_data = data.loc['2019-08-23':'2019-09-21']

def forcast(input_data):
    ress = []
    # 预测后8天
    for i in tqdm.tqdm(range(30)):
        if not ress:
            ress.append(get_sample_prediction(input_data, dp_model.predict_theta_from_input))
        else:
            ress.append(get_sample_prediction(np.concatenate((input_data.tail(30-i), np.c_[np.array(ress),np.arange(400,400+i,1)]), axis=0),
                                              dp_model.predict_theta_from_input))
    return ress


# for i in tqdm.tqdm(range(20)):
#     ress.append(get_sample_prediction(predict_data[0][i:i+30], dp_model.predict_theta_from_input))

# batch = ts.next_batch(1, 30)
#
ress = []
for i in tqdm.tqdm(range(301)):
    ress.append(get_sample_prediction(predict_data, dp_model.predict_theta_from_input))

tot_res = pd.DataFrame(ress)
tot_res = tot_res.T
# res_df = pd.DataFrame()
# for i in range(100):
#     res_df = pd.concat([res_df, pd.DataFrame(forcast(predict_data)).T])
# tot_res = res_df.T.tail(7).reset_index(drop=True)

# plt.plot(batch[1].reshape(1), linewidth=6)
tot_res['predict'] = tot_res[np.where(tot_res.apply(np.sum)==np.percentile(tot_res.apply(np.sum), 50))[0][0]]
tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
#
# plt.plot(res_df.T.tail(7).reset_index(drop=True), label='predict')
plt.plot(data.loc['2019-06-06':'2019-07-05'].reset_index(drop=True)['order'], label='true')
plt.plot(tot_res.predict, label='predict')
plt.plot(tot_res.mu, linewidth=2, label='mu')
plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
plt.title('Prediction uncertainty %s' % 'B007SIR08C-A23TNQB4GVF91M-ATVPDKIKX0DER.csv')

plt.legend()
plt.show()