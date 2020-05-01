from argparse import ArgumentParser, Namespace
import torch
from solver_vaegan import Solver_vaegan
from solver_patchgan import Solver_patchgan
import yaml 
import sys
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AdaVAE_d_2')
    parser.add_argument('-vctk_dir', '-vctk', default='/home/mlpjb04/Voice_Conversion/Corpus-VCTK/wav48')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-logdir', default='log/')
    
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')

    parser.add_argument('-store_model_path', default='model_patchVAE/')
    parser.add_argument('-load_model_path', default='model/')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=10000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=200000, type=int)

    parser.add_argument('--use_vaegan', action='store_true')
    parser.add_argument('--use_patchgan', action='store_true')


    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    os.makedirs(args.store_model_path,exist_ok=True)
    
    if args.use_vaegan:
        solver = Solver_vaegan(config=config, args=args)

    if args.use_patchgan:
        solver = Solver_patchgan(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters,)
