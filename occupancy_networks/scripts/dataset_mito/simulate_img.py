import sys
from glob import glob
from multiprocessing import Pool
from functools import partial
import os
import ipdb
import numpy as np
import multiprocessing
import cv2
import tqdm
import math as m
import importlib.util
import random
# 
import skimage
import skimage.measure
from PIL import Image
import argparse
from skimage.transform import downscale_local_mean
from tifffile import imwrite

MAX_XY=2000     #nm
MIN_XY=-2000
MIN_Z=0
MAX_Z=1400

def read_conf(conf_file, para ):
    assert  os.path.exists(conf_file)
    spec = importlib.util.spec_from_file_location(para, conf_file)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[para] = foo
    # ipdb.set_trace()
    spec.loader.exec_module(foo)
    return foo


too_big_list = []

def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`
  https://stackoverflow.com/questions/6802577/rotation-of-3d-vector'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': return np.dot(X, np.array([
    [1.,  0,  0],
    [0 ,  c, -s],
    [0 ,  s,  c]
  ]))
  elif axis == 'y': return np.dot(X, np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
  ]))
  elif axis == 'z': return np.dot(X, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
  ]))

# def process_matrix(locations, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp, wvl=0.510):
#     result = np.zeros((locations.shape[0], size_x * size_x))
#     for i in range(locations.shape[0]):
#         if(i % 100 == 0):
#             print(i)
#         psf = msPSF.gLXYZParticleScan(
#             mp, step_size_xy, psf_size_x, np.array(locations[i, 2]),
#             zv=stage_delta, wvl=wvl, normalize=False,
#             px=locations[i, 0], py=locations[i, 1])
#         psf_sampled = downscale_local_mean(psf, (1, sampling, sampling))
#         result[i, :] = psf_sampled.ravel()
#     return result

def process_matrix_all_z(locations, size_x, step_size_xy, psf_size_x, stage_delta, sampling, mp, args, wvl=0.510, ):
    # ipdb.set_trace()
    if args.microscope_type == 'confocal': 
        from simulation import microscPSFmod_conf as msPSF
    else:
        from simulation import microscPSFmod_epi as msPSF
    psf = msPSF.gLXYZParticleScan(
        mp, step_size_xy, psf_size_x, locations[:,2],
        zv=stage_delta, wvl=wvl, normalize=False,
        px=locations[:, 0], py=locations[:, 1])
    return np.reshape(psf, (psf.shape[0], -1))

def save_physics_gt(particlesArray, number, pixel_size, image_size, max_xy, save_fname, args =None):
    # ipdb.set_trace()
    particlesArray=np.delete(particlesArray,2,1)
    particlesArray*=1000
    particlesArray=particlesArray
    particlesArray = particlesArray+(2*max_xy)

    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    if pimage.shape[0] < particlesArray.shape[0] :
        print('Here oo oo')
        return 0
        # ipdb.set_trace()
    pimage[particlesArray[:,1].astype(int),particlesArray[:,0].astype(int)]=255
    pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)

    img = Image.fromarray(pimage2)
    img=img.convert('RGB')
    img.save(save_fname)
    return 1

def brightness_trace(t_off, t_on, rate, frames):

    T = np.zeros((2, frames))
    T[0, :] = np.random.exponential(scale=t_off, size=(1, frames))
    T[1, :] = np.random.exponential(scale=t_on, size=(1, frames))

    B = np.zeros((2, frames))
    B[1, :] = rate * T[1, :]

    T_t = T.ravel(order="F")
    B_t = B.ravel(order="F")

    T_cum = np.cumsum(T_t)
    B_cum = np.cumsum(B_t)

    start = np.random.randint(0, 10)

    br = np.diff(np.interp(np.arange(start, start + frames + 1), T_cum, B_cum))
    #br = 100
    return br

def generate_mitochondria(in_file, args):

    np.random.seed(0)
    print(in_file)

    # # Output parameters
    # size_x = 128  # size of x/y in pixels (image)
    # size_t = 1 # number of acquisitions

    # System parameters
    # step_size_xy = 0.042 # [um] pixel size
    mic_type = args.microscope_type
    if mic_type == 'confocal':
        from simulation import microscPSFmod_conf as msPSF
    else:
        from simulation import microscPSFmod_epi as msPSF
    mp = read_conf(args.mic_conf, 'm_params').m_params
    sim_p = read_conf(args.mic_conf, 'sim_params').sim_params

    nlow=np.random.randint(sim_p.n_low_min, sim_p.n_low_max, 1)
    nhigh=np.random.randint(sim_p.n_high_min, sim_p.n_high_max ,1)
    max_xy = int(1000* 32* sim_p.step_size_xy)
    stage_delta = -1  # [um] negative means toward the objective
    # mp = msPSF.m_params  # Default parameters
    sampling = 1
    psf_size_x = sampling * sim_p.size_x
    # resolution = 24

    # Fluorophore parameters
    t_off = 0 # changed by krishna, needs to be checked, original value 8
    t_on = 1 # changed by krishna, needs to be checked, original value 2
    rate = 10

    number = in_file.split('/')[-1].split('.')[0]  #.split('_')[1]#.
    pc_data = np.load(in_file)
    pred_sclaing_term = 1
    # pred_sclaing_term = 150
    particles = pc_data.f.points * pred_sclaing_term
    n_particles = len(particles)
    print(number, 'number of emitters', n_particles)
    center =  np.mean(particles, axis = 0) # - [(MIN_XY+MAX_XY)/2,(MIN_XY+MAX_XY)/2,(MIN_Z + MAX_Z)/2]
    # particlesArray_orig = (particles - center) * args.resolution   
    particlesArray_orig = (particles - center) * args.resolution * 0.001 # Empiar 10791 at 24nm

    new_list = []
    seen = set()
    for emi in particlesArray_orig:
        t = tuple(emi)
        if t not in seen:
            new_list.append(emi)
            seen.add(t)
    particlesArray_orig = np.asarray(new_list)
    n_particles = len(particlesArray_orig)
    # DONT simulate stacks with this file !!!!!!!!!!!!!!!!!
    for z_tr in sim_p.z_transes:
        particlesArray_orig[:, 2] = particlesArray_orig[:,2] + z_tr
        # ipdb.set_trace()
        brightness = np.zeros((n_particles, sim_p.size_t))
        for i in range(n_particles):
            brightness[i, :] = brightness_trace(t_off, t_on, rate, sim_p.size_t)
        # ipdb.set_trace()    
        for angl in sim_p.angles:
            for ax in sim_p.axes:
                # ipdb.set_trace()
                if angl == 0:
                    particlesArray = particlesArray_orig.copy()
                    if ax == 'x':
                        ax = ''
                    else:
                        continue
                else:
                    if ax == 'z' and angl == m.pi: 
                        continue
                    particlesArray = rotate(particlesArray_orig, angl, axis=ax )
                particlesArray_gt = particlesArray.copy()

                angl_deg = str(int(m.degrees(angl)))
                # if not args.write_gt:
                if z_tr == 0: 
                    save_fname = os.path.join(args.img_folder, str(number)+ '_'+ ax + angl_deg +'.tif')
                    save_noise_fname = os.path.join(args.img_folder, 'N_'+str(number) +'_'+ ax+ angl_deg + '.tif')
                    save_noise_param = os.path.join(args.img_folder, 'N_'+str(number) + '_'+ ax+ angl_deg + '.txt')
                elif z_tr > 0:
                    save_fname = os.path.join(args.img_folder, str(number)+ '_'+ ax + angl_deg +'_p.tif')
                    save_noise_fname = os.path.join(args.img_folder, 'N_'+str(number) +'_'+ ax+ angl_deg + '_p.tif')
                    save_noise_param = os.path.join(args.img_folder, 'N_'+str(number) + '_'+ ax+ angl_deg + '_p.txt')
                else:
                    save_fname = os.path.join(args.img_folder, str(number)+ '_'+ ax + angl_deg+'_n.tif')
                    save_noise_fname = os.path.join(args.img_folder, 'N_'+str(number) +'_'+ ax+ angl_deg + '_n.tif')
                    save_noise_param = os.path.join(args.img_folder, 'N_'+str(number) + '_'+ ax+ angl_deg + '_n.txt')

                # save_fname = os.path.join(args.img_folder, str(number)+ '_'+ ax + angl_deg+'.tif')
                if not args.overwrite and os.path.exists(save_fname):
                    print('Image already exist: %s' % save_fname)
                    continue  
                
                # Get particles data 
                particlesData = process_matrix_all_z(particlesArray, 
                sim_p.size_x, sim_p.step_size_xy, psf_size_x, stage_delta, 
                sampling, mp, args, wvl=sim_p.wvl,)

                # Image generation
                image = np.zeros((sim_p.size_t, sim_p.size_x, sim_p.size_x))
                for t in range(sim_p.size_t):
                    b = brightness[:, t]
                    image[t, :, :] = np.reshape(
                        np.sum(particlesData * b[:, None], axis=0), (sim_p.size_x, sim_p.size_x))
                # if not args.write_gt:
                import matlab.engine
                eng = matlab.engine.start_matlab()
                sim_path = eng.genpath('dataset_mito/simulation')
                eng.addpath(sim_path, nargout=0)
                # ipdb.set_trace() 
                
                for t in range(sim_p.size_t):
                    d=image[t, :, :]
                    d/=np.max(d)
                    d*=255
                    d=d.astype(np.uint16)                   
                    if args.write_gt:
                        # ipdb.set_trace()
                        save_gtname= os.path.join(args.gt_folder, str(number) +'_'+ ax+ angl_deg + '.png')
                        wrote_gt = save_physics_gt(particlesArray_gt, number, int(1000*sim_p.step_size_xy), 
                                    sim_p.size_x, max_xy, save_gtname, args)
                        if not wrote_gt : # todo: replace last 3 parameters
                            print("too big mito", in_file)
                            too_big_list.append(in_file)
                            return
                    else:
                        outF=open(save_noise_param,'w')
                        outF.write(str(nlow)+','+str(nhigh))
                        outF.close()
                    imwrite(save_fname,d,compress=6)
                    print('Wrote ', save_fname, angl_deg, ax)
                    # print(save_fname,save_noise_fname)
                    x=eng.add_poisson_noise_image(save_fname,int(nhigh),int(nlow),save_noise_fname)  #sending input to the function
        

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--resolution', type=int, default=24,
                    help='Resolution of EM segmaentation.')
parser.add_argument('--write_gt', type=bool, default=False,
                    help='Whether to wirite GT.')
parser.add_argument('--microscope_type', type =str, default='epi', 
                    help='Type of micrsocope being simulated')
parser.add_argument('--mic_conf', type =str, default='epi', 
                    help='Parameters of micrsocope being simulated')
parser.add_argument('--img_folder', type=str,
                    help='Output path for simulated images.')
parser.add_argument('--gt_folder', type=str,
                    help='Output path for simulated gt.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')



def main(args):
    input_files = glob(os.path.join(args.in_folder, '*.npz'))
    n_images = len(input_files)
    print('Number of files', n_images)


    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(generate_mitochondria, args=args), input_files)
    else:
        for p in input_files[::]:
            generate_mitochondria(p, args)
    print('Too big list', too_big_list)
if __name__ == '__main__':
    args = parser.parse_args()
    if args.microscope_type =='confocal':
        msPSF = map(__import__, "simulation.microscPSFmod_conf.")
    else:
        msPSF = map(__import__, 'simulation.microscPSFmod_epi')
    main(args)

