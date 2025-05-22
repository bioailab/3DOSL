import os 
import pickle
import numpy as np
# import cv2

def write_pickle(data, filename):
    with open(filename, "wb") as f_out:
        pickle.dump(data, f_out)

def mkdir_if_not_existing(dir_path):
    if not (os.path.exists(dir_path)):
        os.mkdir(dir_path)

def does_not_exist(file_path):
    return not (os.path.isfile( file_path))

def resize_voxel(voxel, sz=64):
    output = np.zeros((sz, sz, sz), dtype=np.uint8)

    if np.argmax(voxel.shape) == 0:
        for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, sz)):
            output[i] = cv2.resize(voxel[int(s)], (sz, sz)) 
    elif np.argmax(voxel.shape) == 1:
        for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, sz)):
            output[:, i] = cv2.resize(voxel[:, int(s)], (sz, sz))
    elif np.argmax(voxel.shape) == 2:
        for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, sz)):
            output[:, :, i] = cv2.resize(voxel[:, :, int(s)], (sz, sz))

    return output

def str2slices(s):
    s = s [6:-1]        # Remove the slice and brackets
    start, stop, step = s.replace(' ', '').split(',')
    if  step == 'None': 
        step = None
    else:
        step = int(step)
    start = int(start)
    stop = int(stop)
    return slice(start, stop, step)
