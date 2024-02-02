"""***********************************************************************
FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition

-------------------------------------------------------------------------
File: environment.py

Version: 1.0
***********************************************************************"""


import numpy as np
from mysql import CryptoData, fetch_dataset, StockData
from typing import List, Optional, Union, Tuple


class MarketEnv:
    def __init__(self, date_from: str, date_to: str, type_: str, window_size: int, reb_freq: int,
                 split_ratio: List[int], use_short: bool = False, add_cash: bool = True,
                 normalize: Optional[str] = None, use_partial: Optional[int] = None,
                 f_l: float = 0.0033, f_s: float = 0.004):
        """
        The market environment of various types of assets.

        :param date_from: starting date of the data
        :param date_to: ending date of the data
        :param type_: type of the data
            ex) "Crypto"
        :param tick: data's tick interval in hours
            ex) 1 = 1 hour interval tick data, 24 = 24 hour interval tick data
        :param window_size: total number of tick data for each instance
            ex) 20 = 20 hours worth data for each instance
        :param split_ratio: split ratio for the train/validation/test
        :param use_short: agent can have sell position
        :param add_cash: always consider the first asset of the portfolio as a cash weight
        :param normalize: normalization method for the features X
            Note) all labels Y are represented by the "percent change" which is increase ratio (prev CP vs. next CP)
        :param use_partial: use data partially upto the given number of data
            ex) len(data) = 1,000 -> use_partial = 700 -> len(data) = 700
        :param f_l: Transaction fee for closing long position assets
        :param f_s: Transaction fee for closing short position assets
        """
        self.type_ = type_
        self.data: np.array = None
        self.dates: np.array = None
        self.assets: np.array = None
        self.glob_data: np.array = None

        if self.type_ == 'Crypto':  # 2019-01-01 ~ 2022-11-01
            self.data_cls: CryptoData = CryptoData(usage='get', date_from=date_from, date_to=date_to)
            self.data, self.glob_data, self.dates, self.assets = \
                self.data_cls.fetch_data(to_numpy=True)
        elif self.type_ == 'stocks_ksp':
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='kr', type_='stocks_ksp', to_numpy=True)
        elif self.type_ == 'stocks_index_ksp':  # 2001-01-04 ~ 2022-11-01
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='kr', type_='stocks_ksp', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='kr', type_='index', to_numpy=True)
        elif self.type_ == 'stocks_index_us':  # 1992-08-03 ~ 2022-11-01
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='us', type_='stocks_us', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='us', type_='index', to_numpy=True)
        elif self.type_ == 'stocks_index_cn':  # 2009-01-05 - 2020-12-31 (34)
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='cn', type_='stocks_cn', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='cn', type_='index', to_numpy=True)
        elif self.type_ == 'stocks_index_jp':  # 2013-08-12 ~ 2023-12-19 (118)
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='jp', type_='stocks_jp', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='jp', type_='index', to_numpy=True)
        elif self.type_ == 'stocks_index_uk':  # 2014-01-02 ~ 2023-12-19 (21)
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='uk', type_='stocks_uk', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='uk', type_='index', to_numpy=True)
        else:
            raise ValueError(f"{self.type_}is not supported dataset type.")
        assert np.sum(split_ratio) == 10, 'The ratio sum has to be 10.'

        self.f_l = f_l
        self.f_s = f_s
        self.reb_freq = reb_freq
        self.add_cash = add_cash
        self.use_short = use_short
        self.normalize = normalize
        self.use_partial = use_partial
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.use_glob = False if self.glob_data is None else True

        if self.use_partial is not None:
            self.data = self.data[:, :self.use_partial, :]

        dataset = fetch_dataset(data=self.data, glob_data=None if self.glob_data is None else self.glob_data,
                                window_size=self.window_size, reb_freq=self.reb_freq, split_ratio=self.split_ratio,
                                normalize=self.normalize, dates=self.dates)

        if self.use_glob:
            self.x_trn, self.y_trn, self.x_val, self.y_val, self.x_test, self.y_test, \
                self.d_trn, self.d_val, self.d_test = dataset[0]
            self.glob_x_trn, self.glob_y_trn, self.glob_x_val, self.glob_y_val, self.glob_x_test, self.glob_y_test, \
                _, _, _ = dataset[1]
            self.glob_x, self.glob_y = self.glob_x_trn, self.glob_y_trn
        else:
            self.x_trn, self.y_trn, self.x_val, self.y_val, self.x_test, self.y_test, \
                self.d_trn, self.d_val, self.d_test = dataset

        self.x, self.y = self.x_trn, self.y_trn

        self.state_space: Tuple[int] = self.x[0].shape
        self.action_space: Tuple[int] = self.y[0][0].shape

        self.n_step: int = 0
        self.mode: str = 'train'
        self.total_step: int = self.x.shape[0]
        self.state: np.array = self.x[self.n_step]
        self.w = np.array([[1.0] + [0.0] * self.action_space[0]]) if self.add_cash \
            else np.array([[0.0] * self.action_space[0]])
        self.glob_state: np.array = self.glob_x[self.n_step] if self.use_glob else None

        self._print_statistics()

    def step(self, action: np.array) -> \
            Tuple[np.array, Union[np.array, np.float64], bool, Optional[np.array], float, float, np.array, np.array]:
        # Step function here can be assumed to take action with size 1 batch only i.e., (B, N+1) = (1, N+1)
        s_, done = self.x[self.n_step + 1], False
        g_ = self.glob_x[self.n_step + 1] if self.use_glob else None

        if not self.add_cash:
            t_action = action  # No truncation required
        else:
            t_action = action[:, 1:]  # Truncated without cash = v_{t,\0} (1, N)

        tot_fee = self.calc_tot_fee(W=self.w[0], V=action[0])  # W=(1, N+1), V=(1, N+1), where 1 is size of the batch

        # Calc w(=_w), w'(=w), and profit & loss (pl)
        w_cash = np.expand_dims(action[:, 0], 1)  # Cash weights
        cr_o = self.y[self.n_step][0] / 100  # (N, )
        cr_h = self.y[self.n_step][1] / 100  # (N, )
        d = (1 + np.sign(t_action)) / 2 * t_action * cr_o + \
            (1 - np.sign(t_action)) / 2 * ((cr_h >= 1.0) * np.abs(t_action) + (cr_h < 1.0) * np.abs(t_action) * cr_o)

        pl = np.sum(np.sign(t_action) * d, axis=1)[0]
        if not self.add_cash:
            _w = t_action + d
        else:
            _w = np.concatenate([w_cash, t_action + d], axis=1)  # Changed weights
        self.w = _w / np.expand_dims(np.sum(np.abs(_w), axis=1), axis=1)

        r = pl - tot_fee
        self.n_step += 1

        if self.n_step >= self.x.shape[0] - 1:
            done = True

        return s_, r, done, g_, pl, tot_fee, d, cr_o

    def reset(self) -> np.array:
        self.n_step = 0
        self.state = self.x[self.n_step]
        self.glob_state: np.array = self.glob_x[self.n_step] if self.use_glob else None
        self.w = np.array([[1.0] + [0.0] * self.action_space[0]]) if self.add_cash \
            else np.array([[0.0] * self.action_space[0]])
        return self.state, self.glob_state

    def set_mode(self, mode: str):
        self.mode = mode

        if mode == 'train':
            self.x, self.y = self.x_trn, self.y_trn
        elif mode == 'validation':
            self.x, self.y = self.x_val, self.y_val
        elif mode == 'test':
            self.x, self.y = self.x_test, self.y_test

        if self.use_glob:
            if mode == 'train':
                self.glob_x, self.glob_y = self.glob_x_trn, self.glob_y_trn
            elif mode == 'validation':
                self.glob_x, self.glob_y = self.glob_x_val, self.glob_y_val
            elif mode == 'test':
                self.glob_x, self.glob_y = self.glob_x_test, self.glob_y_test

        self.reset()

    def _print_statistics(self):
        print(f"--------------------------Dataset Statistics--------------------------\n"
              f"{'Data normalization:':<20}\t{self.normalize}\n"
              f"{'Train validation ratio:':<20}{self.split_ratio}\n"
              f"{'Number of training steps:':<20}\t{len(self.x_trn) - 1} steps\n"  
              f"{'Number of validation steps:':<20}\t{len(self.x_val) - 1} steps\n"
              f"{'Number of test steps:':<20}\t{len(self.x_test) - 1} steps\n"
              f"{'State Shape':<20}\t{self.state_space}\n"
              f"{'Action Shape':<20}\t{self.action_space}\n"
              f"----------------------------------------------------------------------\n")

    def calc_tot_fee(self, W: np.array, V: np.array) -> float:
        """
        Fixed-point iteration for finding optimal closing weights C* and O*

        :param W: Current portfolio weights
        :param V: Desired portfolio weights

        :return: Total loss weight due to transaction fee occurred during rebalancing it to desired portfolio weights
        """

        N = W.shape[0]
        C = np.zeros(N)
        O = np.zeros(N)
        eps = 1e-10
        sign_x = lambda x: np.sign(np.sign(x) + 0.5)

        while True:
            C_prev = np.copy(C)
            O_prev = np.copy(O)

            A = np.sum((np.abs(C_prev) * ((1+sign_x(C_prev)) * self.f_l / 2 + (1 - sign_x(C_prev)) * self.f_s / 2))[1:])
            C = O_prev + W + V * (A - 1)

            C[W * C < 0] = 0
            C[np.abs(C) > np.abs(W)] = W[np.abs(C) > np.abs(W)]

            A = np.sum((np.abs(C) * ((1 + sign_x(C)) * self.f_l / 2 + (1 - sign_x(C)) * self.f_s / 2))[1:])
            O = (C - W + V * (1 - A))
            O[(W - C) * O < 0] = 0

            if np.linalg.norm(C_prev - C) <= eps:
                break

        return float(A)
