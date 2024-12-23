import copy
import h5py
import math
import numpy as np
import os
import torch

from torch.utils.data import Dataset
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import  random_select_points, shift_point_cloud, jitter_point_cloud, \
    generate_random_rotation_matrix, generate_random_tranlation_vector,generate_random_tranlation_vector1,generate_random_tranlation_vector2, \
    transform, random_crop, shuffle_pc, random_scale_point_cloud, flip_pc,generate_random_rotation_matrix1,generate_random_rotation_matrix2,generate_random_rotation_matrix3


half1 = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
         'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
         'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp']
half1_symmetric = ['bottle', 'bowl', 'cone', 'cup', 'flower_pot', 'lamp']

half2 = ['laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano',
         'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
         'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
half2_symmetric = ['tent', 'vase']


class ModelNet40(Dataset):
    def __init__(self, root, split, npts, p_keep, noise, unseen, ao=False,
                 normal=False):
        super(ModelNet40, self).__init__()
        self.single = False # for specific-class visualization
        assert split in ['train', 'val', 'test']
        self.split = split
        self.npts = npts
        self.p_keep = p_keep
        self.noise = noise
        self.unseen = unseen
        self.ao = ao # Asymmetric Objects
        self.normal = normal
        
        self.half = half1 if split in 'train' else half2
        self.symmetric = half1_symmetric + half2_symmetric
        
        self.label2cat, self.cat2label = self.label2category(
            os.path.join(root, 'shape_names.txt'))
        self.half_labels = [self.cat2label[cat] for cat in self.half]
        self.symmetric_labels = [self.cat2label[cat] for cat in self.symmetric]
        
        
        files = [os.path.join(root, 'ply_data_train{}.h5'.format(i))
                 for i in range(5)]
        if self.split == 'val':
            files = [os.path.join(root, 'ply_data_train{}.h5'.format(4))]
        if self.split == 'test':
            files = [os.path.join(root, 'ply_data_test{}.h5'.format(i))
                     for i in range(2)]

        self.data, self.labels = self.decode_h5(files)
        print(f'split: {self.split}, unique_ids: {len(np.unique(self.labels))}')

        if self.split == 'train':
            self.Rs = [generate_random_rotation_matrix() for _ in range(len(self.data))]
            self.ts = [generate_random_tranlation_vector() for _ in range(len(self.data))]

    def label2category(self, file):
        with open(file, 'r') as f:
            label2cat = [category.strip() for category in f.readlines()]
            cat2label = {label2cat[i]: i for i in range(len(label2cat))}
        return label2cat, cat2label

    def decode_h5(self, files):
        points, normal, label = [], [], []
        for file in files:
            f = h5py.File(file, 'r')
            cur_points = f['data'][:].astype(np.float32)
            cur_normal = f['normal'][:].astype(np.float32)
            cur_label = f['label'][:].flatten().astype(np.int32)
            if self.unseen:
                idx = np.isin(cur_label, self.half_labels)
                cur_points = cur_points[idx]
                cur_normal = cur_normal[idx]
                cur_label = cur_label[idx]
            if self.ao and self.split in ['val', 'test']:
                idx = ~np.isin(cur_label, self.symmetric_labels)
                cur_points = cur_points[idx]
                cur_normal = cur_normal[idx]
                cur_label = cur_label[idx]
            if self.single:
                idx = np.isin(cur_label, [8])
                cur_points = cur_points[idx]
                cur_normal = cur_normal[idx]
                cur_label = cur_label[idx]
            points.append(cur_points)
            normal.append(cur_normal)
            label.append(cur_label)
        points = np.concatenate(points, axis=0)
        normal = np.concatenate(normal, axis=0)
        data = np.concatenate([points, normal], axis=-1).astype(np.float32)
        label = np.concatenate(label, axis=0)
        return data, label

    def compose(self, item, p_keep):
        tgt_cloud = self.data[item, ...]
        keep0 = 0.70
        keep1 = 0.70
        if self.split != 'train':
            np.random.seed(item)
            # if random.random() < 0.9:
            #     R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
            # else:
            R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector2()
            R2, t2 = generate_random_rotation_matrix1(), generate_random_tranlation_vector1()
        else:
            tgt_cloud = flip_pc(tgt_cloud)
            R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
            R1, t1 = generate_random_rotation_matrix2(), generate_random_tranlation_vector2()
            R2, t2 = generate_random_rotation_matrix1(), generate_random_tranlation_vector1()
        src_cloud = random_crop(copy.deepcopy(tgt_cloud), p_keep=keep0)
        src1 = copy.deepcopy(src_cloud)
        src2 = copy.deepcopy(src_cloud)
        src_size = math.ceil(self.npts * keep0)
        tgt_size = self.npts
        if len(p_keep) > 1:
            tgt_cloud = random_crop(copy.deepcopy(tgt_cloud),
                                    p_keep= keep1)
            tgt_size = math.ceil(self.npts * keep1)

        src_cloud_points = transform(src_cloud[:, :3], R, t)
        src_cloud_normal = transform(src_cloud[:, 3:], R)
        src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
                                   axis=-1)
        src_cloud_points1 = transform(src1[:, :3], R1, t1)
        src_cloud_normal1 = transform(src1[:, 3:], R1)
        src_cloud1 = np.concatenate([src_cloud_points1, src_cloud_normal1],
                                   axis=-1)
        
        src_cloud_points2 = transform(src2[:, :3], R2, t2)
        src_cloud_normal2 = transform(src2[:, 3:], R2)
        src_cloud2 = np.concatenate([src_cloud_points2, src_cloud_normal2],
                                   axis=-1)
        src_cloud = random_select_points(src_cloud, m=src_size)
        tgt_cloud = random_select_points(tgt_cloud, m=tgt_size)
        src_cloud1 = random_select_points(src_cloud1, m=src_size)
        src_cloud2 = random_select_points(src_cloud2, m=src_size)
        
        
        src_cloud_R_points = transform(src_cloud[:, :3], np.linalg.inv(R))
        src_cloud_R_normal = transform(src_cloud[:, 3:], np.linalg.inv(R))
        src_cloud_R = np.concatenate([src_cloud_R_points, src_cloud_R_normal],
                                   axis=-1)
        
        src_cloud_R_points1 = transform(src_cloud1[:, :3], np.linalg.inv(R1))
        src_cloud_R_normal1 = transform(src_cloud1[:, 3:], np.linalg.inv(R1))
        src_cloud_R1 = np.concatenate([src_cloud_R_points1, src_cloud_R_normal1],
                                   axis=-1)
        
        src_cloud_R_points2 = transform(src_cloud2[:, :3], np.linalg.inv(R2))
        src_cloud_R_normal2 = transform(src_cloud2[:, 3:], np.linalg.inv(R2))
        src_cloud_R2 = np.concatenate([src_cloud_R_points2, src_cloud_R_normal2],
                                   axis=-1)
      
#        print(src_cloud,src_cloud_R)
        
        if self.split == 'train' or self.noise:
            src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])
            src_cloud_R[:, :3] = jitter_point_cloud(src_cloud_R[:, :3])
            src_cloud1[:, :3] = jitter_point_cloud(src_cloud1[:, :3])
            src_cloud_R1[:, :3] = jitter_point_cloud(src_cloud_R1[:, :3])
            src_cloud2[:, :3] = jitter_point_cloud(src_cloud2[:, :3])
            src_cloud_R2[:, :3] = jitter_point_cloud(src_cloud_R2[:, :3])
            
            tgt_cloud[:, :3] = jitter_point_cloud(tgt_cloud[:, :3])
        tgt_cloud, src_cloud,src_cloud_R = shuffle_pc(tgt_cloud), shuffle_pc(src_cloud),shuffle_pc(src_cloud_R)
        src_cloud1,src_cloud_R1 = shuffle_pc(src_cloud1),shuffle_pc(src_cloud_R1)
        src_cloud2,src_cloud_R2 = shuffle_pc(src_cloud2),shuffle_pc(src_cloud_R2)
#        print(src_cloud.shape,src_cloud_R.shape,tgt_cloud.shape)
        return src_cloud, src_cloud_R,src_cloud1, src_cloud_R1,src_cloud2, src_cloud_R2, tgt_cloud, R, t, R1, t1, R2, t2

    def __getitem__(self, item):
        src_cloud, src_cloud_R,src_cloud1, src_cloud_R1,src_cloud2, src_cloud_R2, tgt_cloud, R, t, R1, t1, R2, t2 = self.compose(item=item,
                                                  p_keep=self.p_keep)
        if not self.normal:
            tgt_cloud, src_cloud,src_cloud_R,src_cloud1,src_cloud_R1,src_cloud2,src_cloud_R2 = tgt_cloud[:, :3], src_cloud[:, :3],src_cloud_R[:, :3],src_cloud1[:, :3],src_cloud_R1[:, :3],src_cloud2[:, :3],src_cloud_R2[:, :3]
            
        x_size = src_cloud.shape[0]
        y_size = tgt_cloud.shape[0]
        if x_size < y_size:
            diff = y_size - x_size  # 计算维度差
            src_cloud = np.concatenate((src_cloud , src_cloud[x_size-1-diff:x_size-1,:]), axis=0)
#            print(src_cloud.shape)
        if x_size > y_size:    
            diff = x_size - y_size  # 计算维度差
            tgt_cloud = np.concatenate((tgt_cloud , tgt_cloud[y_size-1-diff:y_size-1,:]), axis=0) 
        return tgt_cloud, src_cloud, src_cloud_R,src_cloud1, src_cloud_R1, src_cloud2, src_cloud_R2, R, t, R1, t1, R2, t2
    def __len__(self):
        return len(self.data)
