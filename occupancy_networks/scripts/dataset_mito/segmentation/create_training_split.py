import argparse
import os
from os import path as osp
import random
import ipdb
from pathlib import Path
from shutil import copy

parser = argparse.ArgumentParser(
    description='Split data into train, test and validation sets.')
parser.add_argument('--im_folder', type=str,
                    help='Input folder where data is stored.')
parser.add_argument('--out_folder', type=str,
                    help='Input folder where GT is stored.')

parser_nval = parser.add_mutually_exclusive_group(required=True)
parser_nval.add_argument('--n_val', type=int,
                         help='Size of validation set.')
parser_nval.add_argument('--r_val', type=float,
                         help='Relative size of validation set.')

parser_ntest = parser.add_mutually_exclusive_group(required=True)
parser_ntest.add_argument('--n_test', type=int,
                          help='Size of test set.')
parser_ntest.add_argument('--r_test', type=float,
                          help='Relative size of test set.')

parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--seed', type=int, default=4)

def main(args):
    

    # all_samples = [name for name in os.listdir(args.im_folder)
                # if os.path.isdir(os.path.join(args.im_folder, name))]
    all_samples = [name for name in os.listdir(osp.join(args.im_folder,'image'))]
    print('Total samples', len(all_samples))

    if args.shuffle:
        random.shuffle(all_samples)

    # Number of examples
    n_total = len(all_samples)

    if args.n_val is not None:
        n_val = args.n_val
    else:
        n_val = int(args.r_val * n_total)

    if args.n_test is not None:
        n_test = args.n_test
    else:
        n_test = int(args.r_test * n_total)

    if n_total < n_val + n_test:
        print('Error: too few training samples.')
        exit()

    n_train = n_total - n_val - n_test

    assert(n_train >= 0)

    # Select elements
    train_set = all_samples[:n_train]
    val_set = all_samples[n_train:n_train+n_val]
    test_set = all_samples[n_train+n_val:]
    ipdb.set_trace()

    Path( osp.join(args.out_folder, 'image', "train" )).mkdir(parents=True, exist_ok=True)
    Path( osp.join(args.out_folder, 'image', "val" )).mkdir(parents=True, exist_ok=True)
    Path( osp.join(args.out_folder, 'image', "test" )).mkdir(parents=True, exist_ok=True)
    Path( osp.join(args.out_folder, 'segment', "train" )).mkdir(parents=True, exist_ok=True)
    Path( osp.join(args.out_folder, 'segment', "val" )).mkdir(parents=True, exist_ok=True)
    Path( osp.join(args.out_folder, 'segment', "test" )).mkdir(parents=True, exist_ok=True)
     # Copy images
    set_dict  = {"train": train_set , "val": val_set, "test": test_set}
    for dname, dset in set_dict.items():
        for img in dset: 
            copy(osp.join(args.im_folder, 'image' , img) , osp.join(args.out_folder, 'image', dname , img) )
            copy(osp.join(args.im_folder, 'segment' , img) , osp.join(args.out_folder, 'segment', dname , img) )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

"""
Usage: 
export DATA_ROOT='/data_mnt/data/EMPIAR-10791/data/segmentation'
python3 create_training_split.py --im_folder=$DATA_ROOT/data --out_folder=$DATA_ROOT/datatrain \
    --n_val=3000 --n_test=0 --shuffle --seed 0
"""