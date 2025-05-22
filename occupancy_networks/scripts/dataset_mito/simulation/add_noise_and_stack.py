from tifffile import imwrite
import glob
import os
from os import path as osp
import argparse
import matlab.engine
import numpy as np
from pathlib import Path
eng = matlab.engine.start_matlab()
from tqdm import tqdm
import ipdb
import sys
sys.path.append('../')
from utils import read_conf
# sim_path = eng.genpath('dataset_mito/simulation/')
# eng.addpath(sim_path, nargout=0
# save_fname = '/mnt/nas1/apu010/data/EMPIAR-10791/data/no_hole.build/1857/5_multi_view_stack_epi_backup/0_920_0_0.tif'
# save_noise_fname = '/mnt/nas1/apu010/data/EMPIAR-10791/data/no_hole.build/1857/5_multi_view_stack_epi_backup/python.tif'

def main(args):
    mp = read_conf(args.config, 'm_params').m_params
    sim_p = read_conf(args.config, 'sim_params').sim_params
    if args.stack_size == -1:
        num_frames = len(sim_p.z_transes)
    else: 
        num_frames = args.stack_size
    last_frames = glob.glob( osp.join(args.in_folder, '*_*_'+str(num_frames-1)+'.tif' ))
    print('Number of stacks ', len( last_frames))
    # folder_name = "conf2_stack_7"
    
    for frame_n in tqdm(last_frames):
        base_name = osp.basename(frame_n).split('.')[0][:-2]
        save_noise_fname_l = osp.join(args.install_folder, base_name,args.folder_name+'_'+str(num_frames), base_name +'.tif' )
        if (osp.exists(save_noise_fname_l)) and (not args.overwrite ):
            print('Already exxists ', save_noise_fname_l)
            continue
        
        full_stack = []
        save_names = []

        # Create variables and save paths
        for i in range(num_frames-1):
            frame_name = "frame_"+str(i)
            frame_val = frame_n.replace( "_" +str(num_frames-1) +".tif", "_" +str(i) +".tif")
            exec(f"{frame_name} = '{frame_val}'")
            full_stack.append(frame_val)
            # Check f z slice exist
            if not (osp.exists(frame_val) ):
                print('Stack not complete', base_name)
                continue
                        
            # Create save filenames for stacks of size 1, 3, 5, ...
            if not i % 2:
                save_noise_fname = osp.join(args.install_folder, base_name,args.folder_name + '_'+str(i+1))
                Path(save_noise_fname).mkdir(parents=True, exist_ok=True)
                save_noise_fname_name = save_noise_fname+ '/'+base_name+'.tif'

                save_names.append(save_noise_fname_name)
        full_stack.append(frame_n)
        save_names.append(save_noise_fname_l)
        Path(os.path.dirname(save_noise_fname_l)).mkdir(parents=True, exist_ok=True)


        # Generate noise variables
        nlow=matlab.double(list(np.random.randint(sim_p.n_low_min, sim_p.n_low_max, num_frames)))
        nhigh=matlab.double(list(np.random.randint(sim_p.n_high_min, sim_p.n_high_max , num_frames)))
        # ipdb.set_trace()
        ret=np.asarray(eng.add_poisson_noise_stack( full_stack , nhigh, nlow)).astype('uint16')
        z_spacing = sim_p.z_transes[0] - sim_p.z_transes[1]
        step_size_xy = sim_p.step_size_xy
        for i in range(1, num_frames+1, 2):
            imwrite(save_names[i//2], ret[:i], imagej=True, resolution=(1./step_size_xy,1./step_size_xy), metadata= {'spacing':z_spacing, 'unit':'um','axes':'ZYX' } )


        
parser = argparse.ArgumentParser('Add noise and stack z slices.')
parser.add_argument('in_folder', type=str,
                    help='Path to the noiseless images.') 
parser.add_argument('config', 
                    help = 'path to the config file for the simulations')
parser.add_argument('--install_folder', type=str, 
                default= '/data_mnt/data/EMPIAR-10791/data/mitochondria/no_hole.install/1857',
                    help='Path to the noisy images.')
parser.add_argument('--folder_name', default='epi2_multi_img')
parser.add_argument('--stack_size', default = -1)
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)