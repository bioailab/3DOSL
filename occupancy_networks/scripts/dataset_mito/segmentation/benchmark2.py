import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,6,7"
import tensorflow as tf
# tf.enable_eager_execution()
import cv2
import ipdb
import numpy as np
import glob, os
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.keras.metrics import Metric

from sklearn.metrics import jaccard_score

# Mine
DATA_ROOT = '/data_mnt/data/EMPIAR-10791/data/segmentation/benchmark/datatest/'
gtdir=DATA_ROOT+'test_gt/'
resdir=DATA_ROOT+'output_arif_replicate_tr/'
# resdir=DATA_ROOT+'output_new_sim_corr_aug2/'
allf1=0
allf1_1 = 0
count=0
allmiou=0
allprecision=0
allrecall=0
jd=0
gt_zero_count = 0
output_zero= 0

def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

# ipdb.set_trace()
for file in glob.glob(gtdir+"*.png"):
    fName=os.path.basename(file)
    gt=cv2.imread(gtdir+fName,0)
    assert os.path.exists(resdir+fName)
    res=cv2.imread(resdir+fName,0)
    assert gt.shape == res.shape
    if not np.count_nonzero(gt):
        gt_zero_count += 1
        continue
    if not np.count_nonzero(res):
        output_zero += 1  
        # continue
    # assert()
    # gt[gt!=0]=1
    # res[res!=0]=1


    gt[gt!=255]=0
    res[res!=255]=0
    gt[gt==255]=1
    res[res==255]=1
    f1=f1_score(gt, res, average='micro')
    allf1 = allf1 + f1
    #print(f1)
    # f1=f1_score(gt, res, average='macro')
    
    
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(gt, res)
    miou=m.result().numpy()

    y_pred = tf.convert_to_tensor(res, dtype=tf.float32)
    y_true = tf.convert_to_tensor(gt, dtype=tf.float32)
    smooth = 1.
    dc = (dice_coef(y_true, y_pred, smooth=smooth)).numpy()
    allf1_1=allf1_1+dc

    # m = tf.keras.metric.F1score
    
    jac = jaccard_score(res, gt, average='micro')
    # jac = jaccard_score(res, gt, average='micro')

    
    jd=jd+jac
    allmiou=allmiou+miou
    precision,recall,fscore,support=precision_recall_fscore_support(gt, res, average='micro')
    print(fName, miou, dc, f1, precision, recall)
    allprecision=allprecision+precision
    allrecall=allrecall+recall
    count=count+1
    # ipdb.set_trace()
print("f1:"+str(allf1/count))
print("f1_dc:"+str(allf1_1/count))
print("IoU:"+str(jd/count))
print("mIoU:"+str(allmiou/count))
print("precision:"+str(allprecision/count))
print("Recall:"+str(allrecall/count))

print('no of zero out', output_zero)
print('no of zero gt', gt_zero_count)
