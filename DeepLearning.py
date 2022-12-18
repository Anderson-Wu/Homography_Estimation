front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

os.sys.path.append("../")  # Add the project directory
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
try:
    from utils import (make_matching_plot_fast, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


torch.set_grad_enabled(False)


def get_dl_correspondences(img1,img2,outlier):

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError("GPU is required to run this demo.")

    # Initialize LoFTR
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("./weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().to(device=device)



    w1, h1 = img1.shape[1], img1.shape[0]
    w1_new, h1_new = 640,480
    img1 = cv2.resize(img1, (w1_new, h1_new),interpolation=cv2.INTER_AREA).astype('float32')

    frame1_tensor = frame2tensor(img1, device)

    w2, h2 = img2.shape[1], img2.shape[0]
    w2_new, h2_new = 640,480
    img2 = cv2.resize(img2, (w2_new, h2_new),interpolation=cv2.INTER_AREA).astype('float32')

    frame2_tensor = frame2tensor(img2, device)



    vis_range = [0,2000]

    last_data = {'image0':frame1_tensor, 'image1': frame2_tensor}
    matcher(last_data)

    total_n_matches = len(last_data['mkpts0_f'])
    mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
    mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
    mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]

    # Normalize confidence.
    if len(mconf) > 0:
        conf_vis_min = 0.
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

    sort_index = np.argsort(mconf)
    sort_index  = sort_index[::-1]



    points1 = np.array([],dtype=np.float64)
    points2 = np.array([], dtype=np.float64)

    mkpts0 = mkpts0[sort_index]
    mkpts1 = mkpts1[sort_index]


    for i in range(len(mkpts0)):
        points1 = np.insert(points1, len(points1), mkpts0[i][0]/w1_new*w1)
        points1 = np.insert(points1, len(points1), mkpts0[i][1]/h1_new*h1)
        points2 = np.insert(points2, len(points2), mkpts1[i][0]/w2_new*w2 )
        points2 = np.insert(points2, len(points2),mkpts1[i][1]/h2_new*h2)

    points1 = points1.reshape(len(mconf),2)
    points2 = points2.reshape(len(mconf), 2)
    '''
    alpha = 0
    color = cm.jet(mconf, alpha=alpha)
    make_matching_plot_fast(
        img1, img2,mkpts0, mkpts1, color,sort_index)


    if outlier != None:
        for i in outlier:
            points1 = np.delete(points1,i-1,axis=0)
            points2 = np.delete(points2, i-1,axis=0)'''
    return points1,points2






