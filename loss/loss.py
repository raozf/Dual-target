import math
import torch
import torch.nn as nn
from utils import square_dists


def Init_loss(gt_transformed_src, pred_transformed_src, loss_type='mae'):

    losses = {}
    num_iter = 1
    if loss_type == 'mse':
        criterion = nn.MSELoss(reduction='mean')
        for i in range(num_iter):
            losses['mse_{}'.format(i)] = criterion(pred_transformed_src[i],
                                                   gt_transformed_src)
    elif loss_type == 'mae':
        criterion = nn.L1Loss(reduction='mean')
        for i in range(num_iter):
            losses['mae_{}'.format(i)] = criterion(pred_transformed_src[i],
                                                   gt_transformed_src)
    else:
        raise NotImplementedError

    total_losses = []
    for k in losses:
        total_losses.append(losses[k])
    losses = torch.sum(torch.stack(total_losses), dim=0)
    return losses


def compute_loss(ref_cloud, pred_ref_clouds, loss_fn):
    losses = []
    discount_factor = 1
    for i in range(2):
        loss = loss_fn(ref_cloud[..., :3].contiguous(),
                       pred_ref_clouds[i][..., :3].contiguous())
        losses.append(discount_factor**(2 - i)*loss)
    return torch.sum(torch.stack(losses))

def compute_loss1(ref_cloud, pred_ref_clouds, loss_fn):
    loss = loss_fn(ref_cloud[..., :3].contiguous(),
                       pred_ref_clouds[0][..., :3].contiguous())
        
    return loss

def Refine_loss(gt_transformed_src, pred_transformed_src, weights=None, loss_type='mae'):
    losses = {}
    num_iter = len(pred_transformed_src)
    for i in range(num_iter):
        if weights is None:
            losses['mae_{}'.format(i)] = torch.mean(
                torch.abs(pred_transformed_src[i] - gt_transformed_src))
        else:
            losses['mae_{}'.format(i)] = torch.mean(torch.sum(
                weights * torch.mean(torch.abs(pred_transformed_src[i] -
                                               gt_transformed_src), dim=-1)
                / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8), dim=-1))

    total_losses = []
    for k in losses:
        total_losses.append(losses[k])
    losses = torch.sum(torch.stack(total_losses), dim=0)

    return losses


def Ol_loss(x_ol, y_ol, dists):
    CELoss = nn.CrossEntropyLoss()
    x_ol_gt = (torch.min(dists, dim=-1)[0] < 0.05 * 0.05).long() # (B, N)
    y_ol_gt = (torch.min(dists, dim=1)[0] < 0.05 * 0.05).long() # (B, M)
    x_ol_loss = CELoss(x_ol, x_ol_gt)
    y_ol_loss = CELoss(y_ol, y_ol_gt)
    ol_loss = (x_ol_loss + y_ol_loss) / 2
    return ol_loss


def cal_loss(gt_transformed_src, pred_transformed_src,gt_transformed_src_R, pred_transformed_src_R, dists, x_ol, y_ol,
             gt_transformed_src1,gt_transformed_src2,pred_transformed_src1,gt_transformed_src_R1,gt_transformed_src_R2,pred_transformed_src_R1):
    losses = {}
    losses['init1'] = Init_loss(gt_transformed_src,
                               pred_transformed_src[0:1])
    losses['init2'] = Init_loss(gt_transformed_src_R,
                               pred_transformed_src_R[0:1])

    losses['init3'] = Init_loss(gt_transformed_src1,
                               pred_transformed_src1[0:1])
    losses['init4'] = Init_loss(gt_transformed_src_R1,
                               pred_transformed_src_R1[0:1])
    
    losses['init5'] = Init_loss(gt_transformed_src2,
                               pred_transformed_src1[1:2])
    losses['init6'] = Init_loss(gt_transformed_src_R2,
                               pred_transformed_src_R1[1:2])


    if x_ol is not None:
        losses['ol'] = Ol_loss(x_ol, y_ol, dists)
    losses['refine1'] = Refine_loss(gt_transformed_src,
                                   pred_transformed_src[1:],
                                   weights=None)
    losses['refine2'] = Refine_loss(gt_transformed_src_R,
                                   pred_transformed_src_R[1:],
                                   weights=None)
    
    alpha, beta, gamma = 1, 0.10, 1
    if x_ol is not None:
        losses['total'] = 0.5*losses['init1'] + 0.5*losses['init2']+ beta * losses['ol'] + losses['refine1'] + 0.5 * (losses['init3'] + losses['init4']+ losses['init5'] + losses['init6'])
    else:
        losses['total'] = 0.5*losses['init1'] + 0.5*losses['init2']+ losses['refine1'] + 0.5 * (losses['init3'] + losses['init4']+losses['init5'] + losses['init6'])
    return losses

# def cal_loss(gt_transformed_src, pred_transformed_src,gt_transformed_src_R, pred_transformed_src_R, dists, x_ol, y_ol):
#     losses = {}
#     losses['init1'] = Init_loss(gt_transformed_src,
#                                pred_transformed_src[0:1])
#     losses['init2'] = Init_loss(gt_transformed_src_R,
#                                pred_transformed_src_R[0:1])
#     if x_ol is not None:
#         losses['ol'] = Ol_loss(x_ol, y_ol, dists)
#     losses['refine1'] = Refine_loss(gt_transformed_src,
#                                    pred_transformed_src[1:],
#                                    weights=None)   
#     if x_ol is not None:
#         losses['total'] = losses['init1'] + losses['init2']+ 0.1 * losses['ol'] + losses['refine1']
#     else:
#         losses['total'] = losses['init1'] + losses['init2']+ losses['refine1'] 
#     return losses
