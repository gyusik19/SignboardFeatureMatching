from pathlib import Path
from PIL import Image
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
import json
# import dircache

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, idx_pts_bbox, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db_dir', type=str, default='samples/gt/db', help='Path to db dir'
    )
    parser.add_argument(
        '--query_dir', type=str, default='samples/gt/query', help='Path to query dir'
    )
    parser.add_argument(
        '--max_keypoints', type=int, default=1000,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')

    parser.add_argument(
    '--resize', type=int, nargs='+', default=[2000, 1000],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.7,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--k', type=int, default = 4,
        help = 'Box matching threshold'
    )
    opt = parser.parse_args()
    print(opt)

    db_image_list = []
    query_image_list = []
    for item in os.listdir(opt.db_dir):
        file_name, file_extention = os.path.splitext(item)
        if file_extention in '.jpg':
            db_image_list.append(item)
    for item in os.listdir(opt.query_dir):
        file_name, file_extention = os.path.splitext(item)
        if file_extention in '.jpg':
            query_image_list.append(item)
    
    image_pairs = []
    for db_item in db_image_list:
        match_num = db_item.split('@')[0]
        for q_item in query_image_list:
            if q_item.split('@')[0] == match_num:
                image_pairs.append([db_item, q_item])
    

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    db_dir = Path(opt.db_dir)
    query_dir = Path(opt.query_dir)
    k = opt.k
    num_FP, num_TP, num_GT = 0, 0, 0
    db_unmatched, q_unmatched, cnt_matched, cnt_dbsign, cnt_qsign = 0, 0, 0, 0, 0
    # match to unmatch, unmatch to match, unmatch to unmatch
    m2u, u2m, u2u = 0, 0, 0
    TP_list = []
    FP_list = []
    GT_list = []
    for i, pair in enumerate(image_pairs):
        name0, name1 = pair[:2]
        print(name0)
        # db_image, inp0, scales0 = read_image(
        #     db_dir / name0, device, opt.resize, 0, opt.resize_float)
        # query_image, inp1, scales1 = read_image(
        #     query_dir / name1, device, opt.resize, 0, opt.resize_float)
        
        db_boxes = []
        db_gt_fname = name0.split('.')[0] + '.json'
        query_boxes = []
        query_gt_fname = name1.split('.')[0] + '.json'
        with open(db_dir / db_gt_fname, 'r') as f:
            json_data = json.load(f)
            db_boxes = json_data['shapes']
        with open(query_dir / query_gt_fname, 'r') as f:
            json_data = json.load(f)
            query_boxes = json_data['shapes']

        db_image = cv2.imread(str(db_dir / name0), cv2.IMREAD_GRAYSCALE)
        query_image = cv2.imread(str(query_dir / name1), cv2.IMREAD_GRAYSCALE)
        
        # masking out
        db_mask = np.zeros_like(db_image)
        for item in db_boxes:
            min_x, min_y = item['points'][0]
            max_x, max_y = item['points'][2]
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            db_mask = cv2.rectangle(db_mask, (min_x, min_y), (max_x, max_y), (255), -1)
        db_image = cv2.bitwise_and(db_image, db_mask)

        query_mask = np.zeros_like(query_image)
        for item in query_boxes:
            min_x, min_y = item['points'][0]
            max_x, max_y = item['points'][2]
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            query_mask = cv2.rectangle(query_mask, (min_x, min_y), (max_x, max_y), (255), -1)
        query_image = cv2.bitwise_and(query_image, query_mask)

        
        inp0 = frame2tensor(db_image, device)
        inp1 = frame2tensor(query_image, device)
        
        do_match = True
        if do_match:
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
        
        # count signboards in db
        cnt_dbsign += len(db_boxes)
        cnt_qsign += len(query_boxes)

        for item in db_boxes:
            if item['flags']['matched'] == True:
                cnt_matched+=1
        
                

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        print('num points : ', len(mkpts0))
        # for d_idx, db_box in enumerate(db_boxes):
        #     # if db_box['flags']['matched'] == False: continue
        #     db_box_pts = db_box['points']
        #     db_xyxy = (db_box_pts[0][0], db_box_pts[0][1], 
        #             db_box_pts[2][0], db_box_pts[2][1])
        #     pts_in_box = idx_pts_bbox(mkpts0, db_xyxy)
        #     npts = len(pts_in_box)
        #     query_box_matched = -1
        #     max_match = 0
        #     for q_idx, query_box in enumerate(query_boxes):
        #         if query_box['flags']['matched'] == False: continue
        #         query_box_pts = query_box['points']
        #         query_xyxy = (query_box_pts[0][0], query_box_pts[0][1], 
        #                 query_box_pts[2][0], query_box_pts[2][1])
        #         num_matched = len(idx_pts_bbox(mkpts1[pts_in_box], query_xyxy))
        #         if max_match < num_matched:
        #             max_match = num_matched
        #             query_box_matched = q_idx
                    
        #     if query_box_matched != -1 and max_match > 0:
        #         print(db_boxes[d_idx]["label"], query_boxes[query_box_matched]["label"])
        TP_list.append([name0, name1])
        FP_list.append([name0, name1])
        for q_idx, query_box in enumerate(query_boxes):
            # if query_box['flags']['matched'] == False: continue
            # if query_box['flags']['matched'] == True: num_GT += 1
            # if query_box['label'] != "100" and query_box['flags']['changed'] == False: num_GT += 1
            # if query_box['label'] == "100": continue
            if query_box['label'] != "100": num_GT += 1
            query_box_pts = query_box['points']
            query_xyxy = (query_box_pts[0][0], query_box_pts[0][1],
                    query_box_pts[2][0], query_box_pts[2][1])
            pts_in_box = idx_pts_bbox(mkpts1, query_xyxy)
            npts = len(pts_in_box)
            db_box_matched = -1
            max_match = 0
            for d_idx, db_box in enumerate(db_boxes):
                db_box_pts = db_box['points']
                db_xyxy = (db_box_pts[0][0], db_box_pts[0][1], 
                    db_box_pts[2][0], db_box_pts[2][1])
                num_matched = len(idx_pts_bbox(mkpts0[pts_in_box], db_xyxy))
                if max_match < num_matched:
                    max_match = num_matched
                    db_box_matched = d_idx
            if db_box_matched != -1 and max_match >= k :
                # print(db_boxes[db_box_matched]["label"], query_boxes[q_idx]["label"])
                changed = db_boxes[db_box_matched]['flags']['changed']
                db_pred = db_boxes[db_box_matched]["label"]
                q_pred = query_boxes[q_idx]["label"]
                if db_pred == q_pred and (db_pred != "100" and q_pred != "100"):
                    num_TP += 1
                    TP_list.append([db_pred, q_pred])
                else: 
                    num_FP += 1
                    FP_list.append([db_pred, q_pred])
                    if q_pred == "100" and db_pred != "100":
                        u2m += 1
                    elif db_pred == "100" and q_pred != "100":
                        m2u += 1
                    elif db_pred == "100" and q_pred == "100":
                        u2u += 1

    precision = num_TP / (num_TP + num_FP)
    recall = num_TP / num_GT
    print('num GT : ', num_GT)
    print('num TP, num FP', num_TP, num_FP)
    print("recall : ", recall)
    print("precision : ", precision)
    
    db_unmatched = cnt_dbsign - cnt_matched
    q_unmatched = cnt_qsign - cnt_matched
    print('num signs in db : ', cnt_dbsign)
    print('num signs in query : ', cnt_qsign)
    print('num matched : ', cnt_matched)
    print('num unmatched in db : ', db_unmatched)
    print('num unmatched in query : ', q_unmatched)
    print('query unmatched to db matched : ', u2m)
    print('query matched to db unmatched : ', m2u)
    print('query unmatched to db unmatched : ', u2u)
    # print('TP list : ', TP_list)
    # print('FP_list : ', FP_list)