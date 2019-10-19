import numpy as np
from datetime import datetime
from logging import getLogger

logger = getLogger(__name__)


class DataInsufficientError(ValueError):
    pass


class DeepARConf(object):
    def __init__(self,
                 returns_and_dates,
                 model_save_path,
                 is_training=True,
                 batches=8,
                 unroll_steps=64,
                 successive_train_steps_per_epoch=100,  # 'mini-batch'
                 num_time_series=5,
                 rnn_hidden_units=40,
                 rnn_layers=3,
                 learning_rate=1e-5,
                 momentum=0.9,
                 warmup_steps=200,
                 distribution='student-t',
                 degrees_of_freedom=3.):
        """
        Stores metaparameters for a DeepAR model
        :param returns_and_dates: log-returns and dates of the form [covariates as list, log-returns per asset, prices per asset]
        :param model_save_path: absolute path of the saved model (or the path where the model shall be saved)
        :param is_training: Boolean if the model should be run in training mode or not
        :param batches: the number of batches to process simultaneously. in non-training-mode, this is automatically set to 1
        :param unroll_steps: number of unroll steps for the rnn. is set to 1 in non-training mode
        :param num_time_series: number of assets to simulate
        :param rnn_hidden_units: number of hidden units per rnn cell
        :param rnn_layers: number of rnn cells
        :param learning_rate: the learning rate, duh.
        :param momentum: momentum for batch normalization. something between 0 and 1
        :param distribution: distribution. either student-t or normal
        :param degrees_of_freedom: DoF for the student-t distribution. if None, this is predicted as well
        """
        self.returns_and_dates = returns_and_dates

        self.batches = batches
        self.unroll_steps = unroll_steps
        self.successive_train_steps_per_epoch = successive_train_steps_per_epoch
        self.num_covariates = len(returns_and_dates[0][0])
        self.num_time_series = num_time_series

        self.is_training = is_training
        self.rnn_hidden_units = rnn_hidden_units
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.distribution = distribution
        self.df = degrees_of_freedom

        self.warmup_steps = warmup_steps
        self.model_save_path = str(model_save_path)  # convert pathlib.Path to str

        if not is_training:
            self.batches = 1
            self.unroll_steps = 1

    def get_time_series_yielder(self, num_succ_values: int, time_series_start_date: datetime=None):
        """
        yields tuples of the form (current_datetime(as list), input_returns, target_returns, current_time_step_price)
        :param num_succ_values: number of successive tuples that the yielder should yield
        :param time_series_start_date: start date of the yielder. if None, either a random start date is generated
          (if self.training) or the last num_succ_values are returned.
        :return:
        """
        list_representation = []
        for i in range(1, len(self.returns_and_dates)):
            list_representation.append(
                (self.returns_and_dates[i - 1][0],  # datetime as list (== covariates)
                 self.returns_and_dates[i - 1][1],  # current datetime's returns
                 self.returns_and_dates[i][1],  # next datetime's returns
                 self.returns_and_dates[i][~0]))  # next datetime's prices

        # self.batches is ~8 @ training and 1 @ inference
        if time_series_start_date is not None:
            index_of_start_date = np.argwhere([datetime(*(time_step[0])) >= time_series_start_date for time_step in self.returns_and_dates])[0]
            if not index_of_start_date:
                raise DataInsufficientError()
            index_of_unroll_start = index_of_start_date - (self.unroll_steps - 1)
            batch_indices = index_of_unroll_start
        elif self.is_training:  # == self.randomize_time_series:?
            batch_indices = np.random.randint(
                len(list_representation) - self.unroll_steps * (num_succ_values+1),
                size=self.batches)
        else:
            # return the latest entries of list_representation
            batch_indices = np.array([len(list_representation) - self.unroll_steps * num_succ_values]
                                     * self.batches)

        # covariates, prev_samples, target_samples
        # (current_datetime(as list), input_returns, target_returns, current_time_step_price)
        for succ_batch in range(num_succ_values):
            c_t = np.empty((self.batches, self.unroll_steps, self.num_covariates))
            ps_t = np.empty((self.batches, self.unroll_steps, self.num_time_series))
            ts_t = np.empty_like(ps_t)
            cp_t = np.empty_like(ps_t)
            for b in range(self.batches):
                for s in range(self.unroll_steps):
                    c_t[b][s], ps_t[b][s], ts_t[b][s], cp_t[b][s] = list_representation[batch_indices[b] + s]
            yield c_t, ps_t, ts_t, cp_t
            batch_indices += self.unroll_steps