## Dataset

Download [ModelNet40](https://modelnet.cs.princeton.edu) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) [435M].

## Model Training

```
#unseen categories
CUDA_VISIBLE_DEVICES=0 python train.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao --noise --unseen
#unseen shapes
CUDA_VISIBLE_DEVICES=0 python train.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao
```
## Model Evaluation
```
#unseen categories
CUDA_VISIBLE_DEVICES=0 python eval.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao  --unseen --noise  --cuda --checkpoint work_dirs/models/test_min_loss.pth
#unseen shapes
CUDA_VISIBLE_DEVICES=0 python eval.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao  --cuda --checkpoint work_dirs/models/test_min_loss.pth
```

## Registration Visualization

```
#unseen categories
CUDA_VISIBLE_DEVICES=0 python vis.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao  --unseen --noise  --checkpoint work_dirs/models/test_min_loss.pth
#unseen shapes
CUDA_VISIBLE_DEVICES=0 python vis.py --root your_data_path/modelnet40_ply_hdf5_2048/ --ao    --checkpoint work_dirs/models/test_min_loss.pth
```

## Acknowledgements

We thank the authors of [RPMNet](https://github.com/yewzijian/RPMNet), [PCRNet](https://github.com/vinits5/pcrnet_pytorch),and [ROPNet](https://github.com/zhulf0804/ROPNet) for open sourcing their methods. Our model is based on ROPNet, and we have improved by 40%-52% on his basis.  If you are interesting in our method, you can read ROPNet before.   [Point Cloud Registration Using Representative Overlapping Points](https://arxiv.org/abs/2107.02583)
