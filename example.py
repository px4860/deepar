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

file_list = os.listdir('data')

for file_path in file_list:
    data = pd.read_csv(os.getcwd()+'\data\%s' % file_path, header=None, names=['date','order','seller','marketplace'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.drop(columns=['seller','marketplace'], inplace=True)

    ts = TimeSeries(data.head(265))

    # ts = MockTs()
    dp_model = DeepAR(ts, epochs=150)
    dp_model.instantiate_and_fit()


    def get_sample_prediction(sample, fn):
        sample = np.array(sample).reshape(1, 30, 1)
        output = fn([sample])
        samples = []
        # return output[0].reshape(1)
        for mu,sigma in zip(output[0].reshape(1), output[1].reshape(1)):
            samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
        return np.array(samples)


    # predict_data = ts.next_batch(1, 50)[0]
    predict_data = data.loc['2019-08-24':'2019-09-22']

    def forcast(input_data):
        ress = []
        # 预测后8天
        for i in tqdm.tqdm(range(8)):
            if not ress:
                ress.append(get_sample_prediction(input_data, dp_model.predict_theta_from_input))
            else:
                ress.append(get_sample_prediction(np.concatenate((input_data.tail(30-i), np.array(ress)), axis=0),
                                                  dp_model.predict_theta_from_input))
        return ress


    # for i in tqdm.tqdm(range(20)):
    #     ress.append(get_sample_prediction(predict_data[0][i:i+30], dp_model.predict_theta_from_input))

    # batch = ts.next_batch(1, 30)
    #
    # ress = []
    # for i in tqdm.tqdm(range(300)):
    #     ress.append(get_sample_prediction(batch[0], dp_model.predict_theta_from_input))
    res_df = []
    for i in range(100):
        res_df.append(pd.DataFrame(forcast(predict_data)).T)
    tot_res = pd.DataFrame(res_df).T.tail(7)

    # plt.plot(batch[1].reshape(1), linewidth=6)
    tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
    tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
    tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
    tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
    tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
    #
    # plt.plot(res_df.T.tail(7).reset_index(drop=True), label='predict')
    plt.plot(data.tail(7).reset_index(drop=True), label='true')
    plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
    plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
    plt.title('Prediction uncertainty %s' % file_path)

    plt.legend()
    plt.show()