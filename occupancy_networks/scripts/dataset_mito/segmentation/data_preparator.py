import matlab.engine
import argparse
from PIL import Image, ImageOps
import numpy as np
from tifffile import imwrite
import os
import os.path as osp
import glob
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import parser
import ipdb
from tqdm import tqdm

def generate_image(file_list, args):
    nhigh=np.random.randint(160,220)
    nlow=np.random.randint(80,100)

    eng = matlab.engine.start_matlab()
    img=Image.new('RGB', (256,256))
    img_seg=Image.new('RGB', (256,256))
    # count=4*no
    no = file_list[0]
    count = 1
    # if no == 79:
    #     ipdb.set_trace()
    for r in range(2):
        for c in range(2):
            # im1no=count
            # batch=int(file_list[im1no][1])
            gt_name = file_list[count]
            count=count+1
            base_name = gt_name.split('/')[-1].split('.')[0]
            if(osp.isfile(gt_name) and osp.isfile(args.img_folder + '/' + base_name + ".tif" ) ):
                im1 = Image.open(args.img_folder + '/' + base_name + ".tif")
                im3 = Image.open(gt_name)
                img.paste(im1, (r*128, c*128))
                img_seg.paste(im3, (r*128, c*128))
            else:
                print("Path mismatch", gt_name )
                exit()
    img_seg.save(args.out_folder+'/data/segment/'+str(no)+'.png')
    save_fname=args.out_folder+'/data/original/'+str(no)+'.tif'
    img=np.asarray(img)
    img=img[:,:,0]
    img=img.astype(np.uint16)
    imwrite(save_fname,img,compress=6)
    save_noise_fname=args.out_folder+'/data/noisy/'+str(no)+'.tif'
    x=eng.add_poisson_noise_image(save_fname,int(nhigh),int(nlow),save_noise_fname)  #sending input to the function
    img=Image.open(save_noise_fname).convert('LA')
    img.save(args.out_folder+'/data/image/'+str(no)+'.png')

parser = argparse.ArgumentParser('Montage 4 imges and add noise.')
parser.add_argument('--img_folder', type=str,
                    help='path for simulated images.')
parser.add_argument('--gt_folder', type=str,
                    help='Output path for simulated gt.')
parser.add_argument('--out_folder', type=str, 
        default= "/mnt/nas1/apu010/data/EMPIAR-10791/data/segmentation",
                    help='Output path for segmentation taining data.')     
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')               

def main(args):
    Path(args.out_folder + '/data/segment').mkdir(parents=True, exist_ok=True)
    Path(args.out_folder + '/data/original').mkdir(parents=True, exist_ok=True)
    Path(args.out_folder + '/data/noisy').mkdir(parents=True, exist_ok=True)
    Path(args.out_folder + '/data/image').mkdir(parents=True, exist_ok=True)
    np.random.seed(5)#for reproducibility

    gt_pool = glob.glob(args.gt_folder+'/*.png')
    img_pool = []
    for im in gt_pool:
        base_name = im.split('/')[-1].split('.')[0]
        if osp.isfile(args.img_folder + '/' + base_name + ".tif"):
            img_pool.append(im)
    print('Size', len(gt_pool),  len(img_pool))

    total_image_in_a_batch=len(img_pool)#simulation batch size
    total_batch=1#simualtion batch size
    total_image=int(total_image_in_a_batch/4)#
    np.random.shuffle(img_pool)
    file_list = [ img_pool[4*i : 4*(i+1)] for i in range(total_image)]
    file_list = [tuple([i] + x) for i, x in enumerate(file_list) ]

    # file_list = [[[i,j] for i in range (0,total_image_in_a_batch)] for j in range (1, total_batch+1)]
    # file_list=np.squeeze(np.array(file_list))
    # file_list=file_list.reshape(total_image_in_a_batch,2)
    # np.random.shuffle(file_list)
    # ipdb.set_trace()
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(generate_image, args=args), file_list)
    else:    
        for p in tqdm(file_list):
            generate_image(p, args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# for fil in to_be_renamed: 
#     dest = fil.replace('.png', '.tif').replace('5_2d_gt_epi', '5_single_view_epi')  
#     if not osp.isFile(dest):
#         os.move(dest, '/mnt/nas1/apu010/data/EMPIAR-10791/data/no_hole.build/1857/5_scratch')


