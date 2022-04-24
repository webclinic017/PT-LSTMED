from classes import class_ForecastingTrader, class_DataProcessor
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import pickle
import gc

np.random.seed(1)  # NumPy
random.seed(3)  # Python
tf.set_random_seed(2)  # Tensorflow
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

forecasting_trader = class_ForecastingTrader.ForecastingTrader()
data_processor = class_DataProcessor.DataProcessor()

# -------------------------------------UPLOAD--------------------------------------------#
# read prices
df_prices = pd.read_pickle(
    '/content/drive/PairsTrading/2009-2019/commodity_ETFs_intraday_interpolated_screened_no_outliers.pickle')
# df_prices = pd.read_pickle('data/etfs/pickle/commodity_ETFs_intraday_interpolated_screened_no_outliers.pickle')
# split data in training and test
df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                            ('01-01-2009',
                                                             '31-12-2017'),
                                                            ('01-01-2018',
                                                             '31-12-2018'),
                                                            remove_nan=True)
# load pairs
with open('/content/drive/PairsTrading/2009-2019/pairs_unsupervised_learning_optical_intraday.pickle', 'rb') as handle:
    # with open('data/etfs/pickle/2009-2019/pairs_unsupervised_learning_optical_intraday.pickle', 'rb') as handle:
    pairs = pickle.load(handle)
n_years_train = round(len(df_prices_train) / (240 * 78))
print('Loaded {} pairs!'.format(len(pairs)))

# -------------------------------------TRAINING --------------------------------------------#

combinations = [(24, [50])]
hidden_nodes_names = ['50_nodes']

for i, configuration in enumerate(combinations):
    model_config = {"n_in": configuration[0],
                    "n_out": 1,
                    "epochs": 200,
                    "hidden_nodes": configuration[1],
                    "loss_fct": "mse",
                    "optimizer": "rmsprop",
                    "batch_size": 512,
                    "train_val_split": '2017-01-01',
                    "test_init": '2018-01-01'}
    models = forecasting_trader.train_models(pairs, model_config, model_type='rnn')

    # save models for this configuration
    with open('C:/Users/llx/Desktop/pair_trading/codes/PairsTrading-master/forecasting_models/RNN/models_n_in-' + str(configuration[0]) + '_hidden_nodes-' +
              hidden_nodes_names[i] +
              '.pkl', 'wb') as f:
        pickle.dump(models, f)

gc.collect()
