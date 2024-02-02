"""***********************************************************************
FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition

-------------------------------------------------------------------------
File: main.py
- The main python file works with multiple command-line arguments in CLI.

Version: 1.0
***********************************************************************"""


import os
import torch
import socket
import argparse
import numpy as np
import datetime as dt

import utils
from ddpg import DDPG
from environment import MarketEnv
from experiment import Experiment


cur_path = os.path.dirname(os.path.abspath(__file__))
RESULT_CSV_PATH = os.path.join(cur_path, 'out/main_log.csv')
MODEL_SAVE_PATH = os.path.join(cur_path, 'save/')
LOG_PATH = os.path.join(cur_path, 'out/log/')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_arguments():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--process', type=str, default='train')
    parser_.add_argument('--model_name', type=str, default='US-FQ')
    parser_.add_argument('--mu_net', type=str, default='FQNet')

    # Environment settings
    parser_.add_argument('--date_from', type=str, default='1992-08-03')
    parser_.add_argument('--date_to', type=str, default='2022-11-01')
    parser_.add_argument('--type_', type=str, default='stocks_index_us')
    parser_.add_argument('--tick', type=int, default=24)
    parser_.add_argument('--window_size', type=int, default=256)
    parser_.add_argument('--reb_freq', type=int, default=5)
    parser_.add_argument('--normalize', type=str, default='irvpcp')
    parser_.add_argument('--max_portfolio', type=int, default=40)
    parser_.add_argument('--max_epi', type=int, default=150)

    # Ablation model settings
    parser_.add_argument('--use_FRE', type=str2bool, nargs='?', const=True, default=True)
    parser_.add_argument('--use_CTE', type=str2bool, nargs='?', const=True, default=True)

    # Network hyperparameters
    parser_.add_argument('--update_iter', type=int, default=1)
    parser_.add_argument('--mu_lr', type=float, default=5e-05)
    parser_.add_argument('--q_lr', type=float, default=5e-05)
    parser_.add_argument('--lambda_', type=float, default=5e-01)
    parser_.add_argument('--buffer_limit', type=int, default=10000)
    parser_.add_argument('--batch_size', type=int, default=8)
    parser_.add_argument('--tau', type=float, default=5e-02)
    parser_.add_argument('--batch_least_size', type=int, default=500)
    parser_.add_argument('--print_interval', type=int, default=1)
    parser_.add_argument('--pw_dim', type=int, default=5)
    parser_.add_argument('--num_ef', type=int, default=10)
    parser_.add_argument('--dim1', type=int, default=128, help='temp mha dim')
    parser_.add_argument('--dim3', type=int, default=64, help='asset mha dim')
    parser_.add_argument('--var1', type=int, default=4, help='number of frequency state encoder')
    parser_.add_argument('--gpu', type=int, default=2)
    return parser_


if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()

    seed = 0
    utils.fix_seed(seed)

    np.seterr(all='ignore')
    np.set_printoptions(precision=3, linewidth=2000, edgeitems=50, suppress=False)
    torch.set_printoptions(linewidth=2000, edgeitems=50)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.process == 'train':
        new_model_name = str(dt.datetime.now())
        net_settings = (args.pw_dim, args.num_ef, args.dim1, args.dim3, args.var1)
        log_path = f"{LOG_PATH}{new_model_name}_{socket.gethostname()[-1]}_{args.gpu}.log"

        with open(log_path, "a") as f:
            f.write(str(args.__dict__) + "\n")

        env = MarketEnv(date_from=args.date_from, date_to=args.date_to, type_=args.type_,
                        window_size=args.window_size,
                        reb_freq=args.reb_freq, use_short=True, use_partial=None, split_ratio=[7, 2.8, 0.2],
                        normalize=args.normalize, add_cash=True)
        agent = DDPG(env=env, device=device, net_settings=net_settings, mu_net=args.mu_net, mu_lr=args.mu_lr,
                     q_lr=args.q_lr, max_portfolio=args.max_portfolio, mu_target=None, q_target=None,
                     log_path=log_path, use_FRE=args.use_FRE, use_CTE=args.use_CTE)
        utils.summary(networks=[agent.q_target, agent.mu_target])
        max_tr_pv, max_tr_pv_epi, max_val_pv, max_val_sr, max_val_mdd, max_val_pv_epi, reg_term = \
            agent.train(max_epi=args.max_epi, print_interval=args.print_interval, update_iter=args.update_iter,
                        buffer_limit=args.buffer_limit, batch_size=args.batch_size, tau=args.tau,
                        batch_least_size=args.batch_least_size, lambda_=args.lambda_,
                        model_save_path=MODEL_SAVE_PATH + new_model_name + ".pkl")

        out_dict = {'mu_net': args.mu_net, 'max_portfolio': args.max_portfolio, 'window_size': args.window_size,
                    'reb_freq': args.reb_freq, 'max_epi': args.max_epi, 'q_lr': args.q_lr, 'mu_lr': args.mu_lr,
                    'batch_size': args.batch_size, 'tau': args.tau, 'buffer_limit': args.buffer_limit,
                    'max_tr_pv': max_tr_pv, 'max_tr_pv_epi': max_tr_pv_epi, 'max_val_pv': max_val_pv,
                    'max_val_sr': max_val_sr, 'max_val_mdd': max_val_mdd, 'max_val_pv_epi': max_val_pv_epi,
                    'type_': args.type_, 'tick': args.tick, 'dim1': args.dim1, 'dim3': args.dim3,
                    'var1': args.var1, 'use_FRE': 1 if args.use_FRE else 0, 'use_CTE': 1 if args.use_CTE else 0,
                    'reg_term': reg_term}
        _res = dict(dict(args.__dict__, **out_dict), **{'assets': env.assets.tolist()})
        utils.to_csv(_res, RESULT_CSV_PATH)

    elif args.process == 'experiment':
        print(f"Experiment Model: {args.model_name}")
        m_args = {'mu_net': args.mu_net, 'max_portfolio': args.max_portfolio, 'window_size': args.window_size,
                  'reb_freq': args.reb_freq, 'max_epi': args.max_epi, 'q_lr': args.q_lr, 'mu_lr': args.mu_lr,
                  'batch_size': args.batch_size, 'tau': args.tau, 'buffer_limit': args.buffer_limit,
                  'type_': args.type_, 'tick': args.tick, 'dim1': args.dim1, 'dim3': args.dim3,
                  'var1': args.var1, 'use_FRE': 1 if args.use_FRE else 0, 'use_CTE': 1 if args.use_CTE else 0}
        exp = Experiment(model_name=args.model_name, model_save_path=MODEL_SAVE_PATH, m_args=m_args, device=device)
        exp.exp1()

    else:
        raise ValueError(f"Invalid process type \'{args.process}\'")
