from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
from solver_vqvae import Solver_vqvae
from solver_vc import Solver_vc
from solver_pd import Solver_pd
from solver_mbv import Solver_mbv
from solver_norm import Solver_norm
from solver_attn import Solver_attn
import yaml 
import sys
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AutoVC_d')
    parser.add_argument('-vctk_dir', '-vctk', default='/home/mlpjb04/Voice_Conversion/Corpus-VCTK/wav48')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-logdir', default='log/')
    
    parser.add_argument('--use_mbv', action='store_true')
    parser.add_argument('--use_autovc',action='store_true')
    parser.add_argument('--use_vqvae', action='store_true')
    parser.add_argument('--use_pd', action='store_true')
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--use_attn', action='store_true')

    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')

    parser.add_argument('-store_model_path', default='model_selfattn_vqvae128_vae16/')
    parser.add_argument('-load_model_path', default='model/')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=10000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=200000, type=int)

    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    os.makedirs(args.store_model_path,exist_ok=True)

    
    if args.use_vqvae:
        solver = Solver_vqvae(config=config, args=args)

    elif args.use_pd:
        solver = Solver_pd(config=config, args=args)

    elif args.use_autovc:
        solver = Solver_vc(config=config, args=args)

    elif args.use_mbv:
        solver = Solver_mbv(config=config, args=args)

    elif args.use_norm:
        solver = Solver_norm(config=config, args=args)

    elif args.use_attn:
        solver = Solver_attn(config=config, args=args)

    else:
        solver = Solver(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)
