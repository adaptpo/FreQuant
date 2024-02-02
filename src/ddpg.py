"""***********************************************************************
FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition

-------------------------------------------------------------------------
File: ddpg.py

Version: 1.0
***********************************************************************"""


import torch
import random
import collections
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from colorama import Fore as Color, Style

import experiment
from network import FQMuNet, FQQNet


class ReplayBuffer:
    def __init__(self, device, buffer_limit):
        self.device = device
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, w_lst, a_lst, r_lst, s_prime_lst, w_prime_lst, g_lst, g_prime_lst, done_mask_lst = \
            [], [], [], [], [], [], [], [], []

        for transition in mini_batch:
            s, w, a, r, s_prime, w_prime, g, g_prime, done = transition
            s_lst.append(s)
            w_lst.append(w[0])
            a_lst.append(a[0])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            w_prime_lst.append(w_prime[0])
            g_lst.append(g)
            g_prime_lst.append(g_prime)
            done_mask_lst.append([0.0 if done else 1.0])

        is_glob = True if isinstance(g_lst[0], np.ndarray) else False

        return torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(w_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(a_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(w_prime_lst), dtype=torch.float).to(self.device), \
               torch.tensor(np.array(g_lst), dtype=torch.float).to(self.device) if is_glob else None, \
               torch.tensor(np.array(g_prime_lst), dtype=torch.float).to(self.device) if is_glob else None, \
               torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(self,
                 env,
                 device,
                 mu_net=None,
                 q_target=None,
                 mu_target=None,
                 mu_lr=5e-04,
                 q_lr=1e-03,
                 max_portfolio=0,
                 net_settings=None,
                 log_path=None,
                 use_FRE=True,
                 use_CTE=True):
        self.env = env
        self.q_lr = q_lr
        self.mu_lr = mu_lr
        self.device = device
        self.mu_net = mu_net
        self.log_path = log_path
        self.max_portfolio = max_portfolio
        self.settings = (self.env.state_space, self.env.action_space, self.env.use_short, self.env.use_glob,
                         self.max_portfolio)
        self.F, self.A, self.T = self.env.state_space

        self.settings = (*self.settings, use_FRE, use_CTE)
        self.mu_target = FQMuNet(self.settings, net_settings, classifier_=True)
        self.q_target = FQQNet(self.settings, net_settings)

        if mu_target is not None and q_target is not None:
            self.mu_target.load_state_dict(mu_target)
            self.q_target.load_state_dict(q_target)

        self.mu = FQMuNet(self.settings, net_settings, classifier_=True)
        self.q = FQQNet(self.settings, net_settings)

        # if torch.cuda.device_count() > 1:
        #     self.q_target = torch.nn.DataParallel(self.q_target)
        #     self.mu_target = torch.nn.DataParallel(self.mu_target)
        #     self.q = torch.nn.DataParallel(self.q)
        #     self.mu = torch.nn.DataParallel(self.mu)

        self.q_target.to(self.device)
        self.mu_target.to(self.device)
        self.q.to(self.device)
        self.mu.to(self.device)

        self.q.load_state_dict(self.q_target.state_dict())
        self.mu.load_state_dict(self.mu_target.state_dict())

        self.memory = None

    def save_model(self, path: str):
        save_dict = {'q_target': self.q_target.state_dict(), 'mu_target': self.mu_target.state_dict()}
        torch.save(save_dict, path)

    def action(self, s: np.array, w: np.array, g: np.array):
        mu_target_in = {'x': torch.from_numpy(s).float().reshape(1, *self.settings[0]).to(self.device),
                        'w': torch.from_numpy(w).float().to(self.device)}
        if self.env.use_glob:
            mu_target_in['g'] = torch.from_numpy(g).float().reshape(1, self.F, 1, self.T).to(self.device)

        return self.mu_target(**mu_target_in).detach().cpu().numpy()

    def train(self, max_epi=10000, gamma=0.99, batch_size=16,
              buffer_limit=5000, tau=1e-02, print_interval=20, update_iter=2, batch_least_size=100,
              model_save_path='', lambda_=1e-01):

        self.memory = ReplayBuffer(self.device, buffer_limit)
        mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.mu_lr)
        q_optimizer = optim.Adam(self.q.parameters(), lr=self.q_lr)
        max_tr_pv, max_tr_pv_epi, max_val_pv, max_val_sr, max_val_mdd, max_val_pv_epi = 0, 0, 0, 0, 0, 0

        import time
        for n_epi in range(max_epi):
            s_t = time.time()

            self.env.set_mode('train')
            s, g = self.env.reset()
            a_list = []
            done = False
            pv, score = 1.0, 0.0

            while not done:
                in_w = self.env.w
                mu_in = {'x': torch.from_numpy(s).float().reshape(1, *self.settings[0]).to(self.device),
                         'w': torch.from_numpy(in_w).float().to(self.device)}
                if self.env.use_glob:
                    mu_in['g'] = torch.from_numpy(g).float().reshape(1, self.F, 1, self.T).to(self.device)

                a = self.mu(**mu_in)
                a = a.detach().cpu().numpy()

                a_list.append(a)
                s_prime, r, done, g_prime, _, _, _, _ = self.env.step(a)
                self.memory.put((s, in_w, a, r, s_prime, self.env.w, g, g_prime, done))
                score += r
                pv *= (1+r)
                s = s_prime
                g = g_prime
                reg_term = 0

                if self.memory.size() > self.env.total_step * 5:
                    for i in range(update_iter):
                        b_s, b_w, b_a, b_r, b_s_prime, b_w_prime, b_g, b_g_prime, b_done_mask = \
                            self.memory.sample(batch_size)

                        mu_target_in = {'x': b_s_prime, 'w': b_w_prime}
                        if self.env.use_glob:
                            mu_target_in['g'] = b_g_prime

                        q_in = {'x': b_s, 'a': b_a}
                        q_target_in = {'x': b_s_prime, 'a': self.mu_target(**mu_target_in)}
                        if self.env.use_glob:
                            q_in['w'] = b_w
                            q_in['g'] = b_g
                            q_target_in['w'] = b_w_prime
                            q_target_in['g'] = b_g_prime

                        target = b_r + gamma * self.q_target(**q_target_in) * b_done_mask
                        q_loss = F.smooth_l1_loss(self.q(**q_in), target.detach())
                        q_optimizer.zero_grad()
                        q_loss.backward()
                        q_optimizer.step()

                        mu_in = {'x': b_s, 'w': b_w}
                        if self.env.use_glob:
                            mu_in['g'] = b_g
                        q_in = {'x': b_s, 'a': self.mu(**mu_in)}
                        if self.env.use_glob:
                            q_in['w'] = b_w
                            q_in['g'] = b_g

                        # Maximization of q-vals and minimization of regularization term
                        mu_loss = -self.q(**q_in).mean()
                        reg_term = lambda_ * self.mu.get_reg_term(prior_periodicity=[5, 10, 20, 40, 60],
                                                                  prior_glob=True)
                        mu_loss = mu_loss + reg_term
                        mu_optimizer.zero_grad()
                        mu_loss.backward()
                        mu_optimizer.step()

                        _soft_update(self.mu, self.mu_target, tau)
                        _soft_update(self.q, self.q_target, tau)

            if pv >= max_tr_pv:
                max_tr_pv = pv
                max_tr_pv_epi = n_epi

            if n_epi % print_interval == 0 and n_epi != 0:
                _log = f"# of episode: {n_epi}, " \
                       f"reward sum: {score:.3f}, " \
                       f"avg per step: {(score / (self.env.dates.shape[0] - 1)):.3f}, " \
                       f"PV: {(pv * 100):.3f}%, " \
                       f"reg_term: {reg_term:.3f}, " \
                       f"et: {(time.time()-s_t):.3f}s"

                with open(self.log_path, 'a') as f:
                    f.write(_log + "\n")
                print(_log)

                score = 0.0
                val_pv, val_sr, val_mdd = self.test(mode='validation')

                # Save the best model
                if val_pv > max_val_pv:
                    max_val_pv = val_pv
                    max_val_sr = val_sr
                    max_val_mdd = val_mdd
                    max_val_pv_epi = n_epi
                    self.save_model(path=model_save_path)

                    _log = f"New max val pv: {max_val_pv:.3f}"
                    with open(self.log_path, 'a') as f:
                        f.write(_log + "\n")
                    print(_log)

        return max_tr_pv, max_tr_pv_epi, max_val_pv, max_val_sr, max_val_mdd, max_val_pv_epi, \
            reg_term.detach().cpu().item()

    # This can be staticmethod
    def test(self, mode: str):
        self.env.set_mode(mode)
        s, g = self.env.reset()
        done, score, a_hist, r_hist = False, 0.0, [], []

        while not done:
            a = self.action(s, self.env.w, g)
            s_prime, r, done, g_prime, _, _, _, _ = self.env.step(a)
            score += r
            a_hist.append(a[0])
            r_hist.append(r)
            s, g = s_prime, g_prime

        pv = experiment.calc_pv(rewards=np.array(r_hist))
        sr = experiment.calc_sr(rewards=np.array(r_hist), reb_freq=self.env.reb_freq)
        mdd = experiment.calc_mdd(rewards=np.array(r_hist))

        _log = f"{Color.MAGENTA if mode == 'validation' else Color.CYAN}{mode} " \
               f"reward sum: {score:.3f} " \
               f"avg per step: {(score / len(r_hist)):.3f} " \
               f"PV: {(pv * 100):.3f}% " \
               f"SR: {sr:.3f} " \
               f"MDD: {mdd:.3f}{Style.RESET_ALL}"

        with open(self.log_path, "a") as f:
            f.write(_log + "\n")
        print(_log)

        return pv, sr, mdd


def _soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
