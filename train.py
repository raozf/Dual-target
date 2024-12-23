import argparse
import json
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
ROOT = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))

from data import ModelNet40, KittiDataset
from models import ROPNet2,ROPNet
from loss import cal_loss
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc, inv_R_t, batch_transform, setup_seed, square_dists
from configs import train_config_params as config_params

MODE = 'ModelNet40'

test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')


def save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse, global_step, tag,
                 lr=None):
    for k, v in loss_all.items():
        loss = np.mean(v.item())
        writer.add_scalar(f'{k}/{tag}', loss, global_step)
    cur_r_mse = np.mean(cur_r_mse)
    writer.add_scalar(f'RError/{tag}', cur_r_mse, global_step)
    cur_r_isotropic = np.mean(cur_r_isotropic)
    writer.add_scalar(f'rotError/{tag}', cur_r_isotropic, global_step)
    if lr is not None:
        writer.add_scalar('Lr', lr, global_step)
@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, log_freq, writer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    for step, (tgt_cloud,src_cloud,src_cloud_R,src_cloud1,src_cloud_R1,src_cloud2,src_cloud_R2,gtR,gtt,gtR1,gtt1,gtR2,gtt2) in enumerate(tqdm(train_loader)):
        np.random.seed((epoch + 1) * (step + 1))
        tgt_cloud, src_cloud,src_cloud_R, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), src_cloud_R.cuda(),\
                                         gtR.cuda(), gtt.cuda()
        src_cloud1,src_cloud_R1,src_cloud2,src_cloud_R2,gtR1,gtt1,gtR2,gtt2=\
            src_cloud1.cuda(),src_cloud_R1.cuda(),src_cloud2.cuda(),src_cloud_R2.cuda(),gtR1.cuda(),gtt1.cuda(),gtR2.cuda(),gtt2.cuda()
        optimizer.zero_grad()
        results = model(src=src_cloud,
                        src1=src_cloud1,
                        src2=src_cloud2,
                        tgt=tgt_cloud,
                        num_iter=1,
                        train=True)
        pred_Ts = results['pred_Ts']
        pred_src = results['pred_src']
        pred_src_R = results['pred_src_R']
        pred_src2 = results['pred_src2'] 
        pred_src_R2 = results['pred_src_R2'] 
        x_ol = results['x_ol']
        y_ol = results['y_ol']
        inv_R, inv_t = inv_R_t(gtR, gtt)
        inv_R1, inv_t1 = inv_R_t(gtR1, gtt1)
        inv_R2, inv_t2 = inv_R_t(gtR2, gtt2)
        gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                             inv_t)
        gt_transformed_src_R = batch_transform(src_cloud[..., :3], inv_R)
        gt_transformed_src1 = batch_transform(src_cloud1[..., :3], inv_R1,
                                             inv_t1)
        gt_transformed_src_R1 = batch_transform(src_cloud1[..., :3], inv_R1)
        
        gt_transformed_src2 = batch_transform(src_cloud2[..., :3], inv_R2,
                                             inv_t2)
        gt_transformed_src_R2 = batch_transform(src_cloud1[..., :3], inv_R2)



        dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
#        dist2 = square_dists()
        loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                           pred_transformed_src=pred_src,
                           gt_transformed_src_R=gt_transformed_src_R,
                           pred_transformed_src_R=pred_src_R,
                           dists=dists,
                           x_ol=x_ol,
                           y_ol=y_ol,
                           
                           gt_transformed_src1=gt_transformed_src1,
                           gt_transformed_src2=gt_transformed_src2,
                           pred_transformed_src1=pred_src2,
                           gt_transformed_src_R1=gt_transformed_src_R1,
                           gt_transformed_src_R2=gt_transformed_src_R2,
                           pred_transformed_src_R1=pred_src_R2)

        loss = loss_all['total']
        loss.backward()
        optimizer.step()

        R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        global_step = epoch * len(train_loader) + step + 1

        if global_step % log_freq == 0:
            save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                         global_step, tag='train',
                         lr=optimizer.param_groups[0]['lr'])

        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic)
        t_isotropic.append(cur_t_isotropic)
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn, epoch, log_freq, writer):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for step, (tgt_cloud,src_cloud,src_cloud_R,src_cloud1,src_cloud_R1,src_cloud2,src_cloud_R2,gtR,gtt,gtR1,gtt1,gtR2,gtt2) in enumerate(tqdm(test_loader)):
            np.random.seed((epoch + 1) * (step + 1))
            tgt_cloud, src_cloud,src_cloud_R, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), src_cloud_R.cuda(),\
                                         gtR.cuda(), gtt.cuda()
            src_cloud1,src_cloud_R1,src_cloud2,src_cloud_R2,gtR1,gtt1,gtR2,gtt2=\
                src_cloud1.cuda(),src_cloud_R1.cuda(),src_cloud2.cuda(),src_cloud_R2.cuda(),gtR1.cuda(),gtt1.cuda(),gtR2.cuda(),gtt2.cuda()
            results = model(src=src_cloud,
                            src1=src_cloud1,
                            src2=src_cloud2,
                            tgt=tgt_cloud,
                            num_iter=1)
            pred_Ts = results['pred_Ts']
            pred_src = results['pred_src']
            pred_src_R = results['pred_src_R']
            pred_src2 = results['pred_src2'] 
            pred_src_R2 = results['pred_src_R2'] 
            x_ol = results['x_ol']
            y_ol = results['y_ol']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            inv_R1, inv_t1 = inv_R_t(gtR1, gtt1)
            inv_R2, inv_t2 = inv_R_t(gtR2, gtt2)

            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R, inv_t)
            gt_transformed_src_R = batch_transform(src_cloud[..., :3], inv_R)

            gt_transformed_src1 = batch_transform(src_cloud1[..., :3], inv_R1, inv_t1)
            gt_transformed_src_R1 = batch_transform(src_cloud1[..., :3], inv_R1)
        
            gt_transformed_src2 = batch_transform(src_cloud2[..., :3], inv_R2, inv_t2)
            gt_transformed_src_R2 = batch_transform(src_cloud2[..., :3], inv_R2)



            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
    #        dist2 = square_dists()
            loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                           pred_transformed_src=pred_src,
                           gt_transformed_src_R=gt_transformed_src_R,
                           pred_transformed_src_R=pred_src_R,
                           dists=dists,
                           x_ol=x_ol,
                           y_ol=y_ol,
                           
                           gt_transformed_src1=gt_transformed_src1,
                           gt_transformed_src2=gt_transformed_src2,
                           pred_transformed_src1=pred_src2,
                           gt_transformed_src_R1=gt_transformed_src_R1,
                           gt_transformed_src_R2=gt_transformed_src_R2,
                           pred_transformed_src_R1=pred_src_R2)

            loss = loss_all['total']

            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            global_step = epoch * len(test_loader) + step + 1
            if global_step % log_freq == 0:
                save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                             global_step, tag='test')

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params()
    print(args)

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
        with open(os.path.join(args.saved_path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    if MODE == 'ModelNet40':
        train_set = ModelNet40(root=args.root,
                           split='train',
                           npts=args.npts,
                           p_keep=args.p_keep,
                           noise=args.noise,
                           unseen=args.unseen,
                           ao=args.ao,
                           normal=args.normal
                           )
        val_set = ModelNet40(root=args.root,
                          split='val',
                          npts=args.npts,
                          p_keep=args.p_keep,
                          noise=args.noise,
                          unseen=args.unseen,
                          ao=args.ao,
                          normal=args.normal
                          )
        train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    elif MODE == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        train_dataset = KittiDataset(args.root, train_seqs, 2048, 0.3, './data/kitti_list', 'Train', 1.0)
        train_loader = DataLoader(train_dataset,batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=args.num_workers)

        val_seqs = ['08','09','10']
        val_dataset = KittiDataset(args.root, val_seqs, 2048, 0.3, './data/kitti_list','Test', 1.0)

        val_loader = DataLoader(val_dataset,batch_size=args.batchsize, shuffle=False, drop_last=False, num_workers=args.num_workers)

    else:
        raise Exception("not implemented")
    

    model = ROPNet(args)
#    model.load_state_dict(torch.load(args.checkpoint))
    model = model.cuda()
    loss_fn = cal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=40,
                                                                     T_mult=2,
                                                                     eta_min=1e-6,
                                                                     last_epoch=-1)
    writer = SummaryWriter(summary_path)
    for i in tqdm(range(epoch)):
        for _ in train_loader:
            pass
        for _ in val_loader:
            pass
        scheduler.step()
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    for epoch in range(epoch, args.epoches):
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print("当前学习率：", current_lr)
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader=train_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        epoch=epoch,
                                        log_freq=args.log_freq,
                                        writer=writer)
        print_train_info(train_results)
        if  epoch == 0:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "1test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 39:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "40test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 119:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "120test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 279:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "280test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 599:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "600test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 1199:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "1200test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        if  epoch == 1239:
            test_results = test_one_epoch(test_loader=val_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
            print_train_info(test_results)
            saved_path = os.path.join(checkpoints_path, "1240test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
        scheduler.step()   

if __name__ == '__main__':
    main()
