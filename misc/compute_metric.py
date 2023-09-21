import numpy as np
from scipy import spatial as ss

from .utils import hungarian,AverageMeter,AverageCategoryMeter



def compute_metrics(dist_matrix,match_matrix,pred_num,gt_num,sigma):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma
        
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]
    

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]


    return tp,fp,fn



def eval_metrics(pred_data, gt_data_T):
    # print(gt_data_T)
    if gt_data_T['num']>0:
        gt_data = {'num':gt_data_T['num'].numpy().squeeze(), 'points':gt_data_T['points'].numpy().squeeze(),\
                   'sigma':gt_data_T['sigma'].numpy().squeeze()}
    else:
        gt_data = {'num':0, 'points':[],'sigma':[]}

    # print(gt_data)
    tp_s,fp_s,fn_s,tp_l,fp_l,fn_l = [0,0,0,0,0,0]


    if gt_data['num'] ==0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_s = fp_pred_index.shape[0]
        fp_l = fp_pred_index.shape[0]

    if pred_data['num'] ==0 and gt_data['num'] !=0:
        gt_p = gt_data['points']


        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_s = fn_gt_index.shape[0]
        fn_l = fn_gt_index.shape[0]


    if gt_data['num'] !=0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        gt_p = gt_data['points']
        sigma = gt_data['sigma']


        # dist
        dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
        match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

        # sigma_s and sigma_l
        tp_s,fp_s,fn_s = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma)
        tp_l,fp_l,fn_l = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma)
    return tp_s,fp_s,fn_s,tp_l,fp_l,fn_l,





if __name__ == '__main__':
    eval_metrics()
