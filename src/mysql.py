"""***********************************************************************
FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition

-------------------------------------------------------------------------
File: mysql.py
- The utility functions that are incorporated with the connected database
for the information retrieval.

Version: 1.0
***********************************************************************"""


import os
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
from normalizer import normalizer
from typing import List, Optional, Union, Tuple

cur_path = os.path.dirname(os.path.abspath(__file__))
STOCKS_KOSPI_PATH = os.path.join(cur_path, '../data/stocks_ksp.csv')
STOCKS_US_PATH = os.path.join(cur_path, '../data/stocks_us.csv')
INDEX_KOSPI_PATH = os.path.join(cur_path, '../data/index_ksp.csv')
INDEX_NYSE_PATH = os.path.join(cur_path, '../data/index_nyse.csv')
INDEX_NASDAQ_PATH = os.path.join(cur_path, '../data/index_nasdaq.csv')
CRYPTO_PATH = os.path.join(cur_path, '../data/crypto.csv')

# Added Datasets
STOCKS_JP_PATH = '/home/jihyeong/project/DeepPortfolio/data/stocks_jp.csv'
STOCKS_UK_PATH = '/home/jihyeong/project/DeepPortfolio/data/stocks_uk.csv'
STOCKS_CN_PATH = '/home/jihyeong/project/DeepPortfolio/data/stocks_cn.csv'
INDEX_JP_PATH = '/home/jihyeong/project/DeepPortfolio/data/index_jp.csv'
INDEX_UK_PATH = '/home/jihyeong/project/DeepPortfolio/data/index_uk.csv'
INDEX_CN_PATH = '/home/jihyeong/project/DeepPortfolio/data/index_cn.csv'


class StockData:
    def __init__(self, date_to: Optional[str] = None, date_from: Optional[str] = None):
        self.date_from: dt.datetime = dt.datetime.strptime(date_from, '%Y-%m-%d')
        self.date_to: dt.datetime = dt.datetime.strptime(date_to + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
        assert (self.date_to - self.date_from).days >= 0, 'Invalid from or to dates'

    def fetch_data(self, country: str = 'us', type_: str = None, to_numpy: bool = False) -> \
            Tuple[np.array, np.array, np.array]:
        data: Union[pd.DataFrame, np.array] = pd.DataFrame()
        sorted_s_lst: List[str] = []  # Sorted Selection List of stocks
        dates = None
        names = None
        if type_ == 'index':
            path, indices, drop_cols = None, [], []
            if country == 'kr':
                indices.append('ks11')
                path = INDEX_KOSPI_PATH
                drop_cols = ['date']
            elif country == 'us':
                indices.append('nyse')  # By default, it is NYSE Composite
                path = INDEX_NYSE_PATH
                drop_cols = ['date', 'adj_closing_price', 'Volume']
            elif country == 'cn':
                indices.append('sse50')
                path = INDEX_CN_PATH
                drop_cols = ['date']
            elif country == 'jp':
                indices.append('nikkei')
                path = INDEX_JP_PATH
                drop_cols = ['date']
            elif country == 'uk':
                indices.append('ftse100')
                path = INDEX_UK_PATH
                drop_cols = ['date']
            else:
                path = None

            data = pd.read_csv(path, header=0, parse_dates=['date']). \
                sort_values(by=['date'], ascending=True)
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = np.array(indices)
            data.drop(drop_cols, axis='columns', inplace=True)
            sorted_s_lst = indices

        elif type_ == 'stocks_ksp':
            import utils
            data = pd.read_csv(STOCKS_KOSPI_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])
            data = data[data['stock_id'].isin(utils.KOSPI_LST)]

            sorted_s_lst = data[data['date'] == data.iloc[0]['date']]['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id'], axis='columns', inplace=True)

        elif type_ == 'stocks_us':
            data = pd.read_csv(STOCKS_US_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])

            sorted_s_lst = data['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id', 'name'], axis='columns', inplace=True)

        elif type_ == 'stocks_cn':
            data = pd.read_csv(STOCKS_CN_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])
            sorted_s_lst = data[data['date'] == data.iloc[0]['date']]['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id'], axis='columns', inplace=True)

        elif type_ == 'stocks_jp':
            data = pd.read_csv(STOCKS_JP_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])
            sorted_s_lst = data[data['date'] == data.iloc[0]['date']]['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id'], axis='columns', inplace=True)

        elif type_ == 'stocks_uk':
            data = pd.read_csv(STOCKS_UK_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])
            sorted_s_lst = data[data['date'] == data.iloc[0]['date']]['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id'], axis='columns', inplace=True)

        if to_numpy:
            data = data.to_numpy()
            data = data.reshape((-1, len(sorted_s_lst), data.shape[1]))

        return data, dates, names


class CryptoData:
    def __init__(self,
                 usage: str,
                 date_to: Optional[str] = None,
                 date_from: Optional[str] = None):
        self.date_from: dt.datetime = dt.datetime.strptime(date_from, '%Y-%m-%d')
        self.date_to: dt.datetime = dt.datetime.strptime(date_to + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
        assert (self.date_to - self.date_from).days >= 0, 'Invalid from or to date'

        self.usage = usage
        self.main_table: str = 'info'
        self.ticks: List[int] = [1, 2, 4, 8, 12, 24]
        self.sub_tables: List[str] = [f'info_{tick}' for tick in self.ticks]

    def fetch_data(self, to_numpy: bool = False) -> Union[pd.DataFrame, np.array]:
        """
        Fetches the pre-processed crypto data
        """
        data = pd.read_csv(CRYPTO_PATH, header=0, parse_dates=['date']).sort_values(by=['date', 'name'], ascending=[True, True])
        data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
        candidates = data['name'].unique().tolist()
        bitcoin_index = candidates.index('Bitcoin')
        dates = None
        names = None
        glob_data = None

        if to_numpy:
            # Find missing dates with merge -> Impute missing dates with `ffill` -> Drop ['date', 'name']
            pre_set_data = pd.DataFrame(list(product(data['date'].unique(), candidates)), columns=['date', 'name'])
            fill_existing_data = pre_set_data.merge(data, how='left', on=['date', 'name']).\
                sort_values(by=['name', 'date'], ascending=[True, True])
            imp_data = fill_existing_data.\
                fillna(method='ffill').sort_values(by=['date', 'name'], ascending=[True, True]).reset_index(drop=True)

            dates = imp_data['date'].unique()
            names = imp_data['name'].unique()
            imp_data.drop(['date', 'name', 'day_volume', 'capital'], axis='columns', inplace=True)

            imp_data = imp_data.to_numpy()
            data = imp_data.reshape((-1, len(candidates), imp_data.shape[1]))
            glob_data = data[:, [bitcoin_index], :].copy()

        return data, glob_data, dates, names


def fetch_dataset(data: np.array, window_size: int, split_ratio: List[int], normalize: Optional[str] = None,
                  transpose: Optional[Tuple[int, int, int, int]] = None, reb_freq: int = 1,
                  glob_data: Optional[np.array] = None, dates: np.array = None) -> \
        Union[tuple, List[tuple]]:
    """
    Fetches the crypto dataset
    """
    prev_data = np.copy(data)  # preserved data before norm
    prev_glob_data = np.copy(glob_data) if glob_data is not None else None  # preserved data before norm

    def prep(x: np.array, prev_x: np.array, dates: np.array):
        if normalize:
            x, dates = normalizer(features=x, norm_type_=normalize, dates=dates)
        x = x.transpose((2, 1, 0))

        F, N, T = x.shape
        R = T - (window_size + reb_freq + 1)
        assert R > 0, f"Can not make least 1 instance. {T} <= {window_size + reb_freq + 1}"

        X: np.array = None
        Y: np.array = None
        buying_date_indices: list = []

        _t = 0
        while _t + window_size + reb_freq <= T:
            buying_date_indices.append(_t + window_size)
            _x = x[:, :, _t:_t + window_size]
            X = _x if X is None else np.vstack((X, _x))

            _b_p = prev_x[_t + window_size, :, 0]  # Buying price (Open)
            _s_p = prev_x[_t + window_size + reb_freq, :, 0]  # Selling price (Open)
            _y_o = (_s_p - _b_p) / _b_p * 100  # cr_o

            # Replace the invalid open change rate due to zero opening price
            _y_o[_y_o == -100.0] = 0.0

            # Highest of the highest price
            _h_p = np.max(prev_x[_t + window_size: _t + window_size + reb_freq, :, 1], axis=0)
            _y_h = (_h_p - _b_p) / _b_p * 100  # cr_h

            Y = np.stack([_y_o, _y_h]) if Y is None else np.vstack((Y, np.stack([_y_o, _y_h])))
            _t = _t + reb_freq

        X = X.reshape((-1, F, N, window_size))
        Y = Y.reshape((-1, 2, N))
        np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if transpose:
            X = X.transpose(transpose)

        _trn_idx = int(len(X) * split_ratio[0] / 10)
        _val_idx = _trn_idx + int(len(X) * split_ratio[1] / 10)

        return X[:_trn_idx], Y[:_trn_idx], X[_trn_idx:_val_idx], Y[_trn_idx:_val_idx], X[_val_idx:], Y[_val_idx:], \
               dates[buying_date_indices[:_trn_idx]], \
               dates[buying_date_indices[_trn_idx:_val_idx]], \
               dates[buying_date_indices[_val_idx:]]

    dataset = prep(data, prev_data, dates)
    glob_dataset = prep(glob_data, prev_glob_data, dates) if glob_data is not None else None

    return [dataset, glob_dataset] if glob_dataset is not None else dataset
