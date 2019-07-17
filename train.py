from datetime import datetime, timedelta
from multiprocessing import Pool
from keras.models import load_model
from keras.models import Model
from keras.utils import np_utils
import time
import numpy as np
import pandas as pd
import keras
import random
from factor_processing_utils import filter_extreme_3sigma_df, neutralization, filter_extreme_3sigma, norm, standardize_series
from neural_utils import LossHistory
from keras.layers import Conv2D, LSTM, Concatenate, Dense, Dropout, BatchNormalization, Input
from keras import layers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

# In[ ]:
# 因子读取
def get_features():
    feature_list = {}
    with pd.HDFStore('../AShare/valuation', 'r') as data:
        cap = data['market_cap']
        cap.index = pd.to_datetime(cap.index)
        cap = cap.loc[start_date:end_date]
        feature_list['cap'] = cap
        pb = data['pb_ratio']
        pb.index = pd.to_datetime(pb.index)
        pb = pb.loc[start_date:end_date]
        feature_list['pb'] = pb
        pe = data['pe_ratio']
        pe.index = pd.to_datetime(pe.index)
        pe = pe.loc[start_date:end_date]
        feature_list['pe'] = pe
        ps = data['ps_ratio']
        ps.index = pd.to_datetime(ps.index)
        ps = ps.loc[start_date:end_date]
        feature_list['ps'] = ps
    with pd.HDFStore('../AShare/industry', 'r') as data:
        industry_code = data['industry_code']
        industry_code.index = pd.to_datetime(industry_code.index)
        industry_code = industry_code.loc[start_date:end_date]
    with pd.HDFStore('../AShare/primary', 'r') as data:
        close = data['close'].loc[start_date:end_date]
        adjfactor = data['adj_factor'].loc[start_date:end_date]
        close_adj = close*adjfactor
    with pd.HDFStore('../AShare/indicator', 'r') as data:
        for key in data.keys():
            try:
                temp = data[key].astype(np.float64)
                if temp.index[0] == datetime.strptime("2016-01-04", "%Y-%m-%d"):
                    temp.index = pd.to_datetime(temp.index)
                    feature_list.append(temp)
                else:
                    print(key)
            except:
                print(key)
# In[ ]:
if __name__ == "__main__":
    start_date = '2015-01-05'
    end_date = '2019-05-10'
    feature_dict = get_features()
