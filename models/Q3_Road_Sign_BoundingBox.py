#!/usr/bin/env python
# coding: utf-8

# In[1]:


#library imports
import os
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# In[2]:


path_img = Path('Q3_data/images')
path_ann = Path('Q3_data/annotations')


# In[3]:


def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

def Img_df (path_ann):
    annotations = filelist(path_ann, '.xml')
    ann_list = []
    for path_ann in annotations:
        root = ET.parse(path_ann).getroot()
        ann = {}
        ann['filename'] = Path(str(path_img) + '/'+ root.find("./filename").text)
        ann['xmin'] = int(root.find("./object/bndbox/xmin").text)
        ann['ymin'] = int(root.find("./object/bndbox/ymin").text)
        ann['xmax'] = int(root.find("./object/bndbox/xmax").text)
        ann['ymax'] = int(root.find("./object/bndbox/ymax").text)
        ann_list.append(ann)
    return pd.DataFrame(ann_list)


# In[4]:


df_train = Img_df(path_ann)


# In[5]:


#Reading an image
def read_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[6]:


def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[2],x[1],x[4],x[3]])


# In[7]:


def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)


# In[8]:


#Populating Training DF with new paths and bounding boxes
new_paths = []
new_bbs = []
train_path_resized = Path('Q3_data/images_resized')
for index, row in df_train.iterrows():
    new_path,new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),30)
    new_paths.append(new_path)
    new_bbs.append(new_bb)
df_train['new_path'] = new_paths
df_train['new_bb'] = new_bbs


# In[9]:


X = []
for i in range(0, 877):
    im = read_image(df_train.values[i][5])
    im_v = im.flatten()
    X.append(im_v)
X = np.array(X)
X = np.append(X, np.ones((877,1)), axis = 1)


# In[10]:


Y = df_train['new_bb'].to_numpy().copy()
Z =[]
for i in range(0,877):
    Z.append(Y[i])
Z= np.array(Z)
Y = Z


# In[11]:


def MSE(X, Y, w_opt):
    Y_ = np.matmul(X, w_opt)
    err = 0
    for i in range(0,877):
        err += np.linalg.norm(Y_-Y)
    mse = err/877
    return mse


# In[12]:


def MAE(X, Y, w_opt):
    Y_ = np.matmul(X, w_opt)
    err = 0
    for i in range(0,877):
        err += np.sum(np.absolute(Y_-Y))
    mae = err /(877*4)
    return mae


# In[13]:


def MIOU(X, Y, w_opt):
    Y_ = np.matmul(X, w_opt)
    iou = 0
    for i in range(0,877):
        boxA = Y_[i]
        boxB = Y[i]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou += interArea / float(boxAArea + boxBArea - interArea)	
    miou = iou/877
    return miou


# In[14]:


L = []
MSErr = []
MAErr = []
MeanIOU = []
W=[]
# Ridge Regression 
for lam in range(0, 10):
    w_opt= np.matmul(np.linalg.pinv(np.matmul(X.T,X) - lam * np.eye(3961)),np.matmul(X.T,Y))
    W.append(w_opt)
    mse = MSE(X, Y, w_opt)
    mae = MAE(X, Y, w_opt)
    miou = MIOU(X, Y, w_opt)
    L.append(lam)
    MSErr.append(mse)
    MAErr.append(mae)
    MeanIOU.append(miou)
    print('lambda:',lam, 'MSE:', mse,'MAE:', mae, 'MIOU:', miou)

plt.plot(MSErr)
plt.plot(MAErr)
 
plt.legend(["MSE", "MAE"])
plt.show()

plt.plot(MeanIOU)
plt.legend(["Mean IOU"])
plt.show()


# In[15]:


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


# In[22]:


im = cv2.imread(str(df_train.values[100][5]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[100][6])


# In[23]:


w = np.array(W[0])
Y_ = np.matmul(X, w)
show_corner_bb(im, Y_[100])


# In[30]:


im = cv2.imread(str(df_train.values[98][5]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[98][6])


# In[31]:


w = np.array(W[0])
Y_ = np.matmul(X, w)
show_corner_bb(im, Y_[98])


# In[26]:


im = cv2.imread(str(df_train.values[110][5]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[110][6])


# In[27]:


w = np.array(W[0])
Y_ = np.matmul(X, w)
show_corner_bb(im, Y_[110])


# In[ ]:




