import copy
import glob
import os

import h5py
import numpy as np
import six
from torch.utils.data.dataset import Dataset
import math
from data import mesh
from utils import  random_select_points, shift_point_cloud, jitter_point_cloud, \
    generate_random_rotation_matrix1, generate_random_rotation_matrix2,generate_random_tranlation_vector, \
    transform, random_crop, shuffle_pc, random_scale_point_cloud, flip_pc

def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []
    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


class Scene7(Dataset):
    """ [Scene7 PointCloud](https://github.com/XiaoshuiHuang/fmr) """

    def __init__(self, root, split='test', train=1):
        super().__init__()
        if train > 0:
            pattern = '*.ply'
        elif train == 0:
            pattern = '*.ply'
        else:
            pattern = ['*.ply', '*.ply']
        if split == "test":
            classes = ["7-scenes-office"]
        else:
            classes = ["7-scenes-chess", "7-scenes-fire", "7-scenes-heads", "7-scenes-pumpkin",
                       "7-scenes-redkitchen", "7-scenes-stairs"]
        rootdir = os.path.join(root, '7scene')
        if isinstance(pattern, six.string_types):
            pattern = [pattern]
        # find all the class names
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # get all the 3D point cloud paths for the class of class_to_idx
        self.samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not self.samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))
        self.classes = classes
        self.split = split
        self.n_points = 2048
        self.loader = mesh.plyread
        self.noise = False
    def __getitem__(self,index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        """
        path, _ = self.samples[index]
        tgt_cloud = np.array(self.loader(path).vertex_array)[:].astype('float32')
        keep0 = 0.70
        keep1 = 0.70
        if self.split != 'train':
            np.random.seed(index)
            R, t = generate_random_rotation_matrix1(), generate_random_tranlation_vector()
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector()
        else:
            R, t = generate_random_rotation_matrix1(), generate_random_tranlation_vector() 
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector()  

        tgt_cloud = random_select_points(tgt_cloud, m=self.n_points)      #2048
                               
        src_cloud = random_crop(copy.deepcopy(tgt_cloud), p_keep=keep0)
        src1 = copy.deepcopy(src_cloud)        
        src_size = math.ceil(self.n_points * keep0)
        tgt_size = self.n_points
        tgt_cloud = random_crop(copy.deepcopy(tgt_cloud),p_keep= keep1)
        tgt_size = math.ceil(self.n_points * keep1)

        src_cloud = transform(src_cloud, R, t)
        src_cloud1 = transform(src1, R1, t1)
        
        src_cloud = random_select_points(src_cloud, m=src_size)
        src_cloud1 = random_select_points(src_cloud1, m=src_size)
        tgt_cloud = random_select_points(tgt_cloud, m=tgt_size)
        
        src_cloud_R = transform(src_cloud, np.linalg.inv(R))
        src_cloud_R1 = transform(src_cloud1, np.linalg.inv(R1))
        if self.split == 'train' or self.noise:
            src_cloud = jitter_point_cloud(src_cloud)
            src_cloud_R = jitter_point_cloud(src_cloud_R)
            src_cloud1 = jitter_point_cloud(src_cloud1)
            src_cloud_R1 = jitter_point_cloud(src_cloud_R1)
            tgt_cloud = jitter_point_cloud(tgt_cloud)
        tgt_cloud, src_cloud,src_cloud_R = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud),shuffle_pc(src_cloud_R)
        src_cloud1,src_cloud_R1 = shuffle_pc(src_cloud1),shuffle_pc(src_cloud_R1)
#        print(src_cloud.shape,src_cloud_R.shape,tgt_cloud.shape)
        return tgt_cloud, src_cloud, src_cloud_R,src_cloud1, src_cloud_R1, R, t, R1, t1
    def __len__(self):
        return len(self.samples)


class IclNuim(Dataset):
    def __init__(self, root, split='test'):
        super(IclNuim, self).__init__()
        d_path = os.path.join(root, 'icl_nuim', 'train', 'icl_nuim.h5')
        if split == 'test':
            with h5py.File(d_path, 'r') as f:
                self.target = f['points'][...]
                self.target = self.target[1100:,:,:]
#                self.target = self.target[1100:,:,:]
#                print(self.target.shape)
        else:
            with h5py.File(d_path, 'r') as f:
                self.target = f['points'][...]
                self.target = self.target[:1100,:,:]
#                print(self.target.shape)
        self.n_points = 2048
        self.split = split
        self.noise = False
    def __getitem__(self, index):
        keep0 = 0.70
        keep1 = 0.70
        if self.split == 'test':
            np.random.seed(index)
            R, t = generate_random_rotation_matrix1(), generate_random_tranlation_vector()
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector()
        else:
            R, t = generate_random_rotation_matrix1(), generate_random_tranlation_vector() 
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector() 
            
        tgt_cloud = self.target[index][:].astype('float32')
        tgt_cloud = random_select_points(tgt_cloud, m=self.n_points)      #2048
        
        src_cloud = random_crop(copy.deepcopy(tgt_cloud), p_keep=keep0)
        src1 = copy.deepcopy(src_cloud)
        src_size = math.ceil(self.n_points * keep0)
        tgt_size = self.n_points
        tgt_cloud = random_crop(copy.deepcopy(tgt_cloud),p_keep= keep1)
        tgt_size = math.ceil(self.n_points * keep1)
        src_cloud = transform(src_cloud, R, t)
        src_cloud1 = transform(src1, R1, t1)

        src_cloud = random_select_points(src_cloud, m=src_size)
        src_cloud1 = random_select_points(src_cloud1, m=src_size)
        tgt_cloud = random_select_points(tgt_cloud, m=tgt_size)   
        src_cloud_R = transform(src_cloud, np.linalg.inv(R))
        src_cloud_R1 = transform(src_cloud1, np.linalg.inv(R1))     
        if self.split == 'train' or self.noise:
            src_cloud = jitter_point_cloud(src_cloud)
            src_cloud_R = jitter_point_cloud(src_cloud_R)
            src_cloud1 = jitter_point_cloud(src_cloud1)
            src_cloud_R1 = jitter_point_cloud(src_cloud_R1)
            tgt_cloud = jitter_point_cloud(tgt_cloud)
        tgt_cloud, src_cloud,src_cloud_R = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud),shuffle_pc(src_cloud_R)
        src_cloud1,src_cloud_R1 = shuffle_pc(src_cloud1),shuffle_pc(src_cloud_R1)
#            print(src_cloud.shape,src_cloud_R.shape,tgt_cloud.shape)
        return tgt_cloud, src_cloud, src_cloud_R,src_cloud1, src_cloud_R1, R, t, R1, t1

    def __len__(self):
        return self.target.shape[0]