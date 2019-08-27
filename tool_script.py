import torch
import json
import cv2
import json
import numpy as np
import math
import argparse
import torch
from edgeonly import Model
import ConcaveHull as ch
from torch.utils.data import DataLoader
import operator
import torch.nn as nn
from skimage.io import imsave, imread

def uniformsample_batch(batch, num):
    final = []
    for i in batch:
        i1 = np.asarray(i).astype(int)
        a = uniformsample(i1, num)
        a = torch.from_numpy(a)
        a = a.long()
        final.append(a)
    return final

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def get_hull(edge_logits):
    test = edge_logits
    test_0 = test[:, :]
    test_1 = test[:, :]

    


    # for i in range(len(test_0)):
    #     for j in range(len(test_0[0])):
    #         if test_0[i][j] > 0.7:
    #             test_1[i][j] = 1
    #         else:
    #             test_1[i][j] = 0
    points_pred = []

    for i in range(len(test_1)):
        for j in range(len(test_1[0])):
            if test_1[i][j] > 0:
                points_pred.append([i + 1, j + 1])

    points_pred = np.asarray(points_pred)

    hull = ch.concaveHull(points_pred, 3)
    return hull

def convert_hull_to_cv(hull, bbox):

    original_hull = []

    w = bbox[2] + 30
    h = bbox[3] + 30

    for i in hull:
        original_hull.append([int(i[1] * w / 200), int(i[0] * h / 30)])
    return original_hull

def get_edge(i_url,bbox):
	model_path = './checkpoints_cgcn/epoch15_step11415.pth'

	model = Model(240, 1600, 3).to(device)
	model.load_state_dict(torch.load(model_path)["gcn_state_dict"])
	# if torch.cuda.device_count() > 1:
	#             print("Let's use", torch.cuda.device_count(), "GPUs!")
	#             # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	#             self.model = nn.DataParallel(self.model)
	img = cv2.imread(i_url)

	x0 = max(int(bbox[0]),0)
    y0 = max(int(bbox[1]),0)
    w = max(int(bbox[2]),0)
    h = max(int(bbox[3]),0)

    img = img[y0:y0+h,x0:x0+w]

    img= cv2.copyMakeBorder(img,15,15,15,15,cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (240, 1600))
    img = torch.from_numpy(img)


	img = torch.cat(img)
    img = img.view(-1, 240, 1600, 3)
    img = torch.transpose(img, 1, 3)
    img = torch.transpose(img, 2, 3)
    img = img.float()

	edge_logits, tg2 = model(img.to(device))

	edge_logits1 = edge_logits[0,0,:,:].cpu().numpy()
	arrs1 = np.zeros((30, 200), np.uint8)

	for j in range(len(edge_logits1)):
        for k in range((len(edge_logits1[j]))):
            j1 = math.floor(j)
            k1 = math.floor(k)
            if edge_logits1[j][k]>0.5:
                arrs1[j1][k1]= 255
                # arrs1[j1][k1][1]= 255
                # arrs1[j1][k1][2]= 255
            else:
                arrs1[j1][k1] = 0

    arrs1 = cv2.medianBlur(arrs1,7)
    arrs7 = arrs1
    kernel = np.ones((3,6),np.uint8)
    dilation = cv2.dilate(arrs1,kernel,iterations = 1)
    borders = dilation ^ arrs7

    arrs1 = np.zeros((30, 200), np.float32)
    for j in range(len(edge_logits1)):
        for k in range((len(edge_logits1[j]))):
            j1 = math.floor(j)
            k1 = math.floor(k)
            if borders[j][k] > 180:
                arrs1[j1][k1]= 1.0
                # arrs1[j1][k1][1]= 255
                # arrs1[j1][k1][2]= 255
            else:
                arrs1[j1][k1] = 0.0
    arrs1 = torch.from_numpy(arrs1)
    hull = get_hull(arrs1)

    hull = np.asarray(hull)
    hull = hull.tolist()
    original_hull = convert_hull_to_cv(hull, bbox)

    total_points = 400
    original_hull = uniformsample(np.asarray(original_hull), total_points).astype(int).tolist()
    all_points_x = (original_hull[:,0]/(w+30))*w -15 + x0
    all_points_y = (original_hull[:,1]/(h+30))*h -15 + y0

    return all_points_x, all_points_y