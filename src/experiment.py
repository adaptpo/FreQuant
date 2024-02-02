import torch
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.axes_grid1 import make_axes_locatable

from environment import MarketEnv
from network import FQMuNet

YEAR_TRADING_DAYS_STOCK = 252
YEAR_TRADING_DAYS_CRYPTO = 365
YEAR_RISK_FREE_RATE = 0.025


class Experiment:
    def __init__(self, model_name, model_save_path, device, m_args, env=None):
        """
        This class is used for the purpose of conducting experimental analysis. The results in the paper can be derived
        from using these modules.

        :param model_name: Model name saved in save directory
        :param model_save_path: Saved model path
        :param device: Device type
        :param m_args: Dictionary type model arguments with all the key-value exactly same as the one used at training
        :param env: If environment is given, no need to initialize Market environment again
        """
        self.device = device
        self.model_name: str = model_name
        self.model_save_path: str = model_save_path
        self.m_args: dict = m_args
        self.model_state_dict, _ = load_model(model_name=self.model_name, model_save_path=self.model_save_path)

        for k, v in zip(self.m_args.keys(), self.m_args.values()):
            print(f"{k}: {v}")

        self.add_cash = True
        self.env = MarketEnv(date_from=self.m_args['date_from'],
                             date_to=self.m_args['date_to'],
                             type_=self.m_args['type_'],
                             window_size=self.m_args['window_size'],
                             reb_freq=self.m_args['reb_freq'],
                             use_short=True,
                             use_partial=None,
                             split_ratio=[7, 2.8, 0.2],
                             normalize=self.m_args['normalize'],
                             add_cash=self.add_cash) if env is None else env

        self.F, self.A, self.T = self.env.state_space
        self.settings = (self.env.state_space,
                         self.env.action_space,
                         self.env.use_short,
                         self.env.use_glob,
                         self.m_args['max_portfolio'])
        module_settings = (self.m_args['use_FRE'], self.m_args['use_CTE']) if 'use_FRE' in self.m_args.keys() \
            else (True, True)
        self.settings = (*self.settings, *module_settings)

        self.net_settings = (self.m_args['pw_dim'],
                             self.m_args['num_ef'],
                             self.m_args['dim1'],
                             self.m_args['dim3'],
                             self.m_args['var1'] if 'var1' in self.m_args.keys() else 4)
        self.model = FQMuNet(self.settings, self.net_settings)

        self.model.load_state_dict(self.model_state_dict)
        self.model.to(self.device)
        print(f"Model \"{self.model_name}\" loaded successfully on {self.device}.")

    def get_all_transitions(self, mode: str):
        self.env.set_mode(mode)
        s, g = self.env.reset()

        s_lst, a_lst, r_lst, d_lst, g_lst, pl_lst, tot_fee_lst, diff_lst, cr_o_lst = \
            [], [], [], self.env.d_trn if mode == 'train' else \
            self.env.d_val if mode == 'validation' else self.env.d_test, [], [], [], [], []
        done = False

        while not done:
            mu_in = {'x': torch.from_numpy(s).float().reshape(1, *self.settings[0]).to(self.device),
                     'w': torch.from_numpy(self.env.w).float().to(self.device)}
            if self.env.use_glob and self.m_args['mu_net'] != 'ASMuNet':
                mu_in['g'] = torch.from_numpy(g).float().reshape(1, self.F, 1, self.T).to(self.device)
                g_lst.append(g[3, 0, -1])

            a = self.model(**mu_in).detach().cpu().numpy()
            s_lst.append(s)
            a_lst.append(a[0])

            s_prime, r, done, g_prime, pl, tot_fee, d, cr_o = self.env.step(a)

            r_lst.append(r)
            pl_lst.append(pl)
            tot_fee_lst.append(tot_fee)
            diff_lst.append(d)
            cr_o_lst.append(cr_o)

            s = s_prime
            g = g_prime

        # Retrieved # of dates == # of states, but last state is never visited
        return np.array(s_lst), np.array(a_lst), np.array(r_lst), d_lst[:-1], \
               np.array(g_lst) if self.env.use_glob else None, np.array(pl_lst), np.array(tot_fee_lst), \
               np.array(diff_lst), np.array(cr_o_lst)

    def get_all_details(self, mode='validation'):
        self.env.set_mode(mode)
        s, g = self.env.reset()

        done = False
        temp_attn_scores = []

        while not done:
            mu_in = {'x': torch.from_numpy(s).float().reshape(1, *self.settings[0]).to(self.device),
                     'w': torch.from_numpy(self.env.w).float().to(self.device),
                     'g': torch.from_numpy(g).float().reshape(1, self.F, 1, self.T).to(self.device)}
            a = self.model(**mu_in).detach().cpu().numpy()
            s_prime, _, done, g_prime, _, _, _, _ = self.env.step(a)

            first_FSE_block = self.model.multi_fse_blocks[0]
            # freq_emb = first_FSE_block.freq_emb[0].detach().cpu()
            temp_attn_score = first_FSE_block.temp_attn_score.detach().cpu()
            # asset_attn_score = self.model.asset_attn_score[0].detach().cpu()
            # context_emb = self.model.c[0].detach().cpu()

            temp_attn_scores.append(temp_attn_score)

            s = s_prime
            g = g_prime

        return torch.stack(temp_attn_scores)

    def get_exp_res(self, r_lst):
        fpv = np.prod(r_lst + 1.0)
        arr = calc_arr(type_=self.m_args['type_'], rewards=r_lst, reb_freq=self.m_args['reb_freq'])
        avol = calc_avol(type_=self.m_args['type_'], rewards=r_lst, reb_freq=self.m_args['reb_freq'])
        mdd = calc_mdd(rewards=r_lst)
        asr = calc_asr(type_=self.m_args['type_'], rewards=r_lst, reb_freq=self.m_args['reb_freq'])
        cr = calc_cr(type_=self.m_args['type_'], rewards=r_lst, reb_freq=self.m_args['reb_freq'])
        sor = calc_sor(type_=self.m_args['type_'], rewards=r_lst, reb_freq=self.m_args['reb_freq'])
        print(f"fPV: {fpv}\tARR(%): {arr}\tAVol: {avol}\tASR: {asr}\tSoR: {sor}\tMDD(%): {mdd}\tCR: {cr}")

    def exp1(self):
        s_lst, a_lst, r_lst, d_lst, g_lst, pl_lst, tot_fee_lst, diff_lst, cr_o_lst = \
            self.get_all_transitions(mode='validation')

        print('\t'.join(['cash', *[str(asset_) for asset_ in self.m_args['assets']], 'return']))
        for _idx, action in enumerate(a_lst):
            _format_str = ''
            for asset_weight in action:
                _format_str += f"{(asset_weight*100):.1f}\t" if asset_weight != 0.0 else f"0\t"
            _format_str += f"{(r_lst[_idx] * 100):.1f}"
            print(_format_str)

        plot_action_dist(actions=a_lst)
        self.get_exp_res(r_lst=r_lst)


def calc_pv(rewards: np.array):
    return np.prod(np.array(rewards) + 1)


def calc_sr(rewards: np.array, reb_freq: int) -> float:
    rewards = rewards + 1
    group_size = int(30 / reb_freq)  # Monthly group
    num_group = int(len(rewards) / group_size) if len(rewards) / group_size != 0 else 1
    grouped_reward = np.array_split(rewards, num_group)

    rewards = []
    for reward_group in grouped_reward:
        rewards.append(reward_group.prod())

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    sr = reward_mean / reward_std
    return sr


def get_num_trading_per_year(type_: str, reb_freq: int):
    # Calculate the number of trading days per year for the given type of the market
    if type_ == 'Crypto':
        return int(YEAR_TRADING_DAYS_CRYPTO / reb_freq)
    elif 'stocks' in type_:
        return int(YEAR_TRADING_DAYS_STOCK / reb_freq)
    else:
        raise ValueError(f"Unsupported type: {type_}")


def calc_arr(type_: str, rewards: np.array, reb_freq: int) -> float:
    # Calculate the Annualized Rate of Return (ARR) in Percentage (%)
    return np.mean(rewards * 100) * get_num_trading_per_year(type_=type_, reb_freq=reb_freq)


def calc_avol(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Annualized Volatility (AVol)
    return np.std(rewards) * np.sqrt(get_num_trading_per_year(type_=type_, reb_freq=reb_freq))


def calc_mdd(rewards: np.array) -> float:
    # Calculate the Maximum DrawDown (MDD) in Percentage (%)
    pv, pivot_pv, mdd = 1.0, 1.0, 0.0
    for reward in rewards:
        pv *= (1+reward)
        if pv > pivot_pv:
            pivot_pv = pv
        else:
            mdd = max(mdd, 1-pv/pivot_pv)
    return mdd * 100


def calc_asr(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Annualized Sharpe Ratio (ASR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    avol = calc_avol(type_=type_, rewards=rewards, reb_freq=reb_freq) * 100
    return arr / avol


def calc_cr(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Calmar Ratio (CR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    mdd = calc_mdd(rewards=rewards)
    return arr / mdd


def calc_sor(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Sortino Ratio (SoR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    down_rets = np.minimum(rewards * 100, 0)
    downside_deviation = np.linalg.norm(down_rets - np.mean(down_rets), 2) / np.sqrt(len(down_rets)) * \
                         np.sqrt(get_num_trading_per_year(type_=type_, reb_freq=reb_freq))
    return arr / downside_deviation


def plot_action_dist(actions: np.array, color_range: tuple = None, cmap_: str = 'seismic',
                     bar_orientation='vertical', bar_axis='right'):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(bar_axis, size='5%', pad=0.05)
    if color_range is None:
        im = ax.imshow(actions, cmap=cmap_)
    else:
        im = ax.imshow(actions, cmap=cmap_, vmin=color_range[0], vmax=color_range[1])
    fig.colorbar(im, cax=cax, orientation=bar_orientation)
    # plt.savefig('a_dist', dpi=400)
    plt.show()


def load_model(model_name: str, model_save_path: str):
    checkpoint = torch.load(model_save_path + f"{model_name}.pkl")
    return checkpoint['mu_target'], checkpoint['q_target']