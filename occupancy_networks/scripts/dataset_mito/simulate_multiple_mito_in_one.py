from asyncore import write
from cmath import inf
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
    particlesArray=np.delete(particlesArray,2,1)
    particlesArray*=1000
    particlesArray=particlesArray
    particlesArray = particlesArray+(2*max_xy)

    pimage=np.zeros((pixel_size*image_size,pixel_size*image_size))
    # ipdb.set_trace()

    if pimage.shape[0] < particlesArray.shape[0] :
        print('Here oo oo' ,pimage.shape[0] , particlesArray.shape[0])
        return 0
        # ipdb.set_trace()
    pimage[particlesArray[:,1].astype(int),particlesArray[:,0].astype(int)]=255
    pimage2=skimage.measure.block_reduce(pimage, (pixel_size,pixel_size), np.max)
    #from skimage.transform import downscale_local_mean
    #pimage2=downscale_local_mean(pimage, (100, 100))
    
    
    # print(save_fname)
    #scipy.misc.imsave(save_fname, pimage2)
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

# def generate_mitochondria(batch, name, nlow, nhigh, save_path):
def generate_mitochondria(no, args):
    # print(no)
 
    nlow=np.random.randint(40,70, 1)
    nhigh=np.random.randint(200, 240 ,1)

    save_fname = os.path.join(args.img_folder, str(no)+ '.tif')
    save_noise_fname = os.path.join(args.img_folder, 'N_'+str(no) +'.tif')
    save_noise_param = os.path.join(args.img_folder, 'N_'+str(no) + '.txt')
    np.random.seed(no+70000)
    if not args.overwrite and os.path.exists(save_fname):
        print('Image already exist: %s' % save_fname)
        return  
    # ipdb.set_trace()
    # if no == 8 : 
    #     ipdb.set_trace()
    # Output parameters
    size_x = 128  # size of x/y in pixels (image)
    size_t = 1 # number of acquisitions

    # System parameters
    mic_type = args.microscope_type
    if mic_type == 'confocal':
        step_size_xy = 0.070 # [um] pixel size
        from simulation import microscPSFmod_conf as msPSF
        # For 3d to 2d simulation
        angles= [0, m.pi, m.pi/2.]
        axes = ['x', 'y', 'z', ]
        z_transes = [0.2, -0.2]
    else:
        step_size_xy = 0.080 # [um] pixel size
        z_transes = [0 ]
        wvl = .688 # nm
        from simulation import microscPSFmod_epi as msPSF
        # For 3d segmentation
        angles= [0]+[random.uniform(0, m.pi)]
        axes = ['x']+ [random.choice(['x','y', 'z'])]
        # write_gt = True
    max_xy = int(1000* 32* step_size_xy)
    stage_delta = -1  # [um] negative means toward the objective
    mp = msPSF.m_params  # Default parameters
    sampling = 1
    psf_size_x = sampling * size_x
    # resolution = 24

    # Fluorophore parameters
    t_off = 0 # changed by krishna, needs to be checked, original value 8
    t_on = 1 # changed by krishna, needs to be checked, original value 2
    rate = 10

    emitter_loc = glob(os.path.join(args.in_folder, '*.npz'))
    
    particlesArray = []
    no_of_mito_per_image = 2         # no of mitochondria

    # while()
    mito_count = 0
    while(mito_count < no_of_mito_per_image):
    # for i in range(no_of_mito_per_image):  
        ipdb.set_trace()
        np.random.seed(no)
        file_name = np.random.choice( emitter_loc, 1)[0]        
        pc_data = np.load(file_name)
        particles = pc_data.f.points
        # n_particles = len(particles)
        center =  np.mean(particles, axis = 0) # - [(MIN_XY+MAX_XY)/2,(MIN_XY+MAX_XY)/2,(MIN_Z + MAX_Z)/2]
        # center = [0, 0, 0]
        particlesArray_orig = (particles - center) * args.resolution * 0.001
        ipdb.set_trace()
        sel = np.sort(np.random.choice(len(particles), int(0.1* len(particles))))
        if mito_count > 0 and (len(particlesArray[0]) + len(sel) ) > int(1000*step_size_xy*size_x):
            break
        particlesArray_orig = particlesArray_orig[sel]
        z_tr = random.uniform(0, 1) *.500
        x_tr = random.uniform(-1, 1) * 2
        y_tr = random.uniform(-1, 1) * 2
        # z_tr = 0
        tr = [x_tr, y_tr, z_tr]
        ax = random.choice(['x','y', 'z'])
        angl = random.uniform(0, 1)* m.pi
        # print(no, mito_count , ax, angl, tr, len(sel), file_name)
        # print(no, mito_count , file_name)
        
        particlesArray_orig = rotate(particlesArray_orig, angl, axis=ax )
        particlesArray_orig = particlesArray_orig + tr
        particlesArray.append(particlesArray_orig)
        mito_count +=1
    if mito_count == 0:
        return
    particlesArray = np.vstack(particlesArray)
    print(no, len(particlesArray_orig))
    particlesArray_gt = particlesArray.copy()
    n_particles = len(particlesArray)
    # ipdb.set_trace()
    brightness = np.zeros((n_particles, size_t))
    for i in range(n_particles):
        brightness[i, :] = brightness_trace(t_off, t_on, rate, size_t)    

        # save_fname = os.path.join(args.img_folder, str(number)+ '_'+ ax + angl_deg+'.tif')

            
    # Get particles data 
    particlesData = process_matrix_all_z(particlesArray, 
    size_x, step_size_xy, psf_size_x, stage_delta, 
    sampling, mp, args, wvl=wvl,)

    # Image generation
    image = np.zeros((size_t, size_x, size_x))
    for t in range(size_t):
        b = brightness[:, t]
        image[t, :, :] = np.reshape(
            np.sum(particlesData * b[:, None], axis=0), (size_x, size_x))
    if not args.write_gt:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        sim_path = eng.genpath('dataset_mito/simulation')
        eng.addpath(sim_path, nargout=0)
    # ipdb.set_trace() 
    
    for t in range(size_t):
        d=image[t, :, :]
        d/=np.max(d)
        d*=255
        d=d.astype(np.uint16)                   
        if args.write_gt:
            # ipdb.set_trace()
            save_gtname= os.path.join(args.gt_folder, str(no)  + '.png')
            wrote_gt = save_physics_gt(particlesArray_gt, no, int(1000*step_size_xy), 
                        size_x, max_xy, save_gtname, args)
            if not wrote_gt : # todo: replace last 3 parameters
                # print("too big mito", in_file)
                # too_big_list.append(in_file)
                return
        else:
            outF=open(save_noise_param,'w')
            outF.write(str(nlow)+','+str(nhigh))
            outF.close()
        imwrite(save_fname,d,compress=6)
        print('Wrote ', save_fname)
        # print(save_fname,save_noise_fname)
        # x=eng.add_poisson_noise_image(save_fname,int(nhigh),int(nlow),save_noise_fname)  #sending input to the function


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--n_images', type=int, default=40000, 
                    help = 'number of Images to be created create * 4')
parser.add_argument('--resolution', type=int, default=24,
                    help='Resolution of EM segmaentation.')
parser.add_argument('--write_gt', type=bool, default=False,
                    help='Whether to wirite GT.')
parser.add_argument('--microscope_type', type =str, default='epi', 
                    help='Type of micrsocope being simulated')

parser.add_argument('--img_folder', type=str,
                    help='Output path for simulated images.')
parser.add_argument('--gt_folder', type=str,
                    help='Output path for simulated gt.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')



def main(args):
    input_files = glob(os.path.join(args.in_folder, '*.npz'))
    # n_images = len(input_files)
    n_images = args.n_images
    print('Number of files', n_images)
    from itertools import combinations
    # comb = combinations(input_files, 2)
    # for i in list(comb):
    #     print(i)
    #     break

    # file_1 = np.random.choice(input_files, n_images)
    # file_2 = np.random.choice(input_files, n_images)
    # input_pairs = [ tuple( [file_2[i]]+ [x]) for i,x in enumerate(file_1) ]
    # from itertools import 
    # ipdb.set_trace()
    filenames = range(n_images)

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(generate_mitochondria, args=args), filenames )
    else:
        for p in filenames:
            generate_mitochondria(p, args)
    print('Too big list', too_big_list)
if __name__ == '__main__':
    args = parser.parse_args()
    if args.microscope_type =='confocal':
    # from simulation import microscPSFmod_conf as msPSF
        msPSF = map(__import__, "simulation.microscPSFmod_conf.")
    else:
    # from simulation import microscPSFmod_epi as msPSF
        msPSF = map(__import__, 'simulation.microscPSFmod_epi')
    main(args)

