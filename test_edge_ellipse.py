import torch
import json
import os
import math
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from edgeonly import Model
import torch.nn.utils.rnn as rnn
import edge_imageprovider as image_provider
from tqdm import tqdm
import cv2
from skimage.io import imsave
from functools import reduce
import operator
import math
from GNN import poly_gnn
import copy

import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def angles_in_ellipse(num, a, b):
    angles = np.arange(num)
    angles = angles * 2 * np.pi / num
    angles = angles[::-1]
    return angles

def to_feature(hull, w, h):
    pred_cps = hull
    pred_cps = pred_cps.numpy()

    pred_cps = np.asarray(pred_cps)

    pred_x = pred_cps[:, 0] / w * 120
    pred_y = pred_cps[:, 1] / h * 20

    pred = []
    pred.append((pred_x, pred_y))
    pred = np.asarray(pred[0])
    pred = torch.from_numpy(pred.transpose()).view(1, 960, 2).tolist()

    return pred


def to_binary(hull, w, h):
    pred_cps = hull
    pred_cps = pred_cps.numpy()

    pred_cps = np.asarray(pred_cps)

    pred_x = pred_cps[:, 0] / w * 120
    pred_y = pred_cps[:, 1] / h * 20

    pred = []
    pred.append((pred_x, pred_y))
    pred = np.asarray(pred[0])
    pred = torch.from_numpy(pred.transpose()).view(1, 960, 2).tolist()
    return pred

def sort_points(poly):
    coords = poly[:]
    center = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(coords, key=lambda coord: (-180 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    return  coords


def get_hull(hull, bbox):


    original_hull = hull

    binary_hull = []
    feature_hull = []

    w = bbox[2] + 30
    h = bbox[3] + 30
 
    for i in hull:
        binary_hull.append([i[1] / w, i[0] / h])
        feature_hull.append([i[1] * 20 / w, i[0] * 120 / h])
    return original_hull, binary_hull, feature_hull


def cv_to_py(hull):
    hull = np.asarray(hull)
    opencv_hull = np.zeros((960, 2))
    opencv_hull[:, 0] = hull[:, 1]
    opencv_hull[:, 1] = hull[:, 0]

    return  opencv_hull


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args


def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    val = 'train_val'

    # dataset_train = DataProvider(split='train', opts=opts['train'])
    dataset_test = DataProvider(split=val, opts=opts[val], mode=val)

    # train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
    #   shuffle = True, num_workers=opts['train']['num_workers'], collate_fn=manuscript.collate_fn)

    test_loader = DataLoader(dataset_test, batch_size=opts[val]['batch_size'],
                             shuffle=True, num_workers=opts[val]['num_workers'], collate_fn=image_provider.collate_fn)

    return test_loader


class Tester(object):
    def __init__(self, args, opts):

        self.d_iou1 = dict()
        self.d_iou_c1 = dict()
        self.d_accuracy1 = dict()
        self.d_accuracy_c1 = dict()
        self.total_iou1 = 0
        self.iou1 = 0
        self.total_accuracy1 = 0
        self.accuracy1 = 0
        self.total_count1 = 0
        self.count1 = 0


        self.model_path = './checkpoints_cgcn/epoch10_step10650.pth'
        # self.edge_model_path = 'checkpoints/final_edge_epoch30_step21240.pth'


        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        self.test_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
        # self.model = Model(120, 1200, 3).to(device)
        self.model = Model(160,960,3).to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [20, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(device)


        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])
        # self.model.load_state_dict(torch.load(self.edge_model_path)["state_dict"])

        """
        state_dim is number of input features to fc layer in GCN,
        (number of channels you feed = fdim )
        """
        # state_dim = 303

        # self.gcn_model = poly_gnn.PolyGNN(state_dim=state_dim,
        #                               n_adj=self.opts['n_adj'],
        #                               cnn_feature_grids=self.opts['cnn_feature_grids'],
        #                               coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
        #                               get_point_annotation=self.opts['get_point_annotation'],
        #                               ).to(device)
        # # self.gcn_model.load_state_dict(torch.load(self.model_path)["state_dict"])
        # self.gcn_model.load_state_dict(torch.load(self.model_path, map_location='cpu')["state_dict"])


    def test(self):
        self.model.eval()
        with torch.no_grad():
            f = 0
            for step, data in enumerate(tqdm(self.test_loader)):
                img = data['img']
                img = torch.cat(img)

                img = img.view(-1, 160, 960, 3)
                img = torch.transpose(img, 1, 3)
                img = torch.transpose(img, 2, 3)
                img = img.float()

                # gt_mask = torch.from_numpy(np.asarray(data['poly_mask'])).to(device)

                # bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)

                # hull = torch.from_numpy(np.asarray(data['hull'])).to(device)
                # hull = []

                edge_logits, class_prob = self.model(img.to(device))




                edge_mask = data["edge_mask"]
                vertex_mask = data["vertices_mask"]
                poly_mask = data["poly_mask"]

                w1 = data["w"][0]
                h1 = data["h"][0]


                # bbox = bbox[0].cpu().numpy()

                # batch_ellipse = []
                # for i in range(len(bbox)):
                #     w = bbox[0][2]
                #     h = bbox[0][3]

                #     angles = angles_in_ellipse(100, h, w)
                #     x = w / 2 * np.cos(angles) + w / 2
                #     y = h / 2 * np.sin(angles) + h / 2

                #     x = torch.from_numpy(x)
                #     y = torch.from_numpy(y)

                #     ellipse = torch.stack((x, y))
                #     ellipse = ellipse.permute((1, 0)).numpy()
                #     ellipse = torch.from_numpy(uniformsample(ellipse, 960))

                #     batch_ellipse.append(ellipse)

                # batch_ellipse = torch.stack(batch_ellipse)

                # batch_ellipse = batch_ellipse.tolist()

                # hull_from_data = data['hull']

                # hull_from_data = batch_ellipse

                # feature_hull = []
                # original_hull = []
                # binary_hull = []

                # bbox = data['bbox'][0]

                # w = bbox[2]
                # h = bbox[3]

                # for i in range(edge_logits.shape[0]):
                #     hull_original, hull_binary, hull_feature = get_hull(hull_from_data[i], bbox)
                #     feature_hull.append(hull_feature)
                #     original_hull.append(hull_original)
                #     binary_hull.append(hull_binary)


                # output_dict = self.gcn_model(tg2.to('cpu'), np.asarray(feature_hull), np.asarray(original_hull), np.asarray(binary_hull),
                #                              data['bbox'])

                #
                # binary_hull = to_binary(output_dict['pred_polys'][-1][0], w, h)
                # feature_hull = to_feature(output_dict['pred_polys'][-1][0], w, h)
                #
                #
                #
                #
                #
                # output_dict = self.gcn_model(tg2, np.asarray(feature_hull), np.asarray(original_hull),
                #                              np.asarray(binary_hull),
                #                              data['bbox'])
                #
                # binary_hull = to_binary(output_dict['pred_polys'][-1][0], w, h)
                # feature_hull = to_feature(output_dict['pred_polys'][-1][0], w, h)
                #
                #
                #
                # output_dict = self.gcn_model(tg2, np.asarray(feature_hull), np.asarray(original_hull),
                #                              np.asarray(binary_hull),
                #                              data['bbox'])




            #     loss_sum = 0
            #     pred_cps = output_dict['pred_polys'][-1][0].cpu()
            #     pred_cps = pred_cps.numpy()

            #     dp = data['actual_gt_poly'][0]


            #     pred_cps = np.asarray(pred_cps)



            #     pred_x = pred_cps[:, 0]
            #     pred_y = pred_cps[:, 1]


            #     pred = []
            #     pred.append((pred_y, pred_x))
            #     pred = np.asarray(pred[0])
            #     pred = pred.transpose()

            #     final_pred = copy.deepcopy(pred)

            #     dp = data['actual_gt_poly']

            # # for i in range(len(dp)):
            # #     if clockwise_check(dp[i]) == False:
            # #         idx = [i for i in range(dp[i].size(0)-1, -1, -1)]
            # #         idx = torch.LongTensor(idx)
            # #         dp[i] = dp[i].index_select(0, idx)

            #     dp = uniformsample_batch(dp, 960)

            #     dp = torch.stack(dp)
            #     dp = dp.cpu().numpy()[0]
            #     dp = np.asarray(dp)


            #     #
            #     w = bbox[2] + 20
            #     h = bbox[3]

            #     # def compute_iou_and_accuracy(arrs, edge_mask1):
            #     #     intersection = cv2.bitwise_and(arrs, edge_mask1)
            #     #     union = cv2.bitwise_or(arrs, edge_mask1)

            #     #     intersection_sum = np.sum(intersection)
            #     #     union_sum = np.sum(union)

            #     #     iou = intersection_sum/union_sum

            #     #     total = np.sum(arrs)
            #     #     correct_predictions = intersection_sum

            #     #     accuracy = correct_predictions/total

            #     #     return iou, accuracy

            #     # iou1, accuracy1 = compute_iou_and_accuracy(img, poly_mask1)
            #     # iou2, accuracy2 = compute_iou_and_accuracy(test_mask_directly, poly_mask1)
            #     # print("derived", iou1, accuracy1)
            #     # print("direct", iou2, accuracy2)


            #     #imsave("./test_gcn_pred/" + str(step) + "gt_mask_palm_leaf.jpg", gt_mask)
            #     # img_orig11 = data["img_orig"].cpu().numpy()[0]

            #     reduced_x = pred_x/w*120
            #     reduced_y = pred_y/h*20
            #     #
            #     reduced_coordinates = []
            #     reduced_coordinates.append(reduced_y)
            #     reduced_coordinates.append(reduced_x)
            #     #
            #     reduced_coordinates = np.asarray(reduced_coordinates)
            #     reduced_coordinates = reduced_coordinates.transpose()
            #     final_pred1 = copy.deepcopy(reduced_coordinates)

            #     mask = np.zeros((20, 120))
            #     pred_mask = cv2.fillPoly(mask, np.int32([reduced_coordinates]), [1])
            #     gt_mask = data['poly_mask']

            #     mask77 = np.zeros((h,w),np.float32)
            #     mask77 = cv2.fillPoly(mask77, np.int32([pred]), [1])

            #     # hull4 = data["hull"].cpu().numpy()[0]
            #     # hull4[:,0] = hull4[:,0]
            #     # hull4[:,1] = hull4[:,1]
            #     # mask4 = np.zeros((20, 120))
            #     # pred_mask4 = cv2.fillPoly(mask4, np.int32([hull4]), [1])
            #     # gt_mask4 = data['poly_mask']
                hull = data["hull"].cpu().numpy()[0]
            #     print(hull)

                palm_leaf_pred = copy.deepcopy(data['img_orig'].cpu().numpy()[0])
            # #     # print(palm_leaf_pred)
            # #     # print(data['img_orig'][0].shape)
            # #     # palm_leaf_pred = cv2.resize(data['img_orig'].numpy()[0],(120,20))
                cv2.polylines(palm_leaf_pred, np.int32([hull]), True, [1], thickness=1)
                for point in hull:
                    cv2.circle(palm_leaf_pred, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                imsave("./test_gcn_pred/" + str(step) + "pred_on_palm_leaf.jpg", palm_leaf_pred)
            #     imsave("./test_gcn_pred/" + str(step) + "mask_palm_leaf.jpg", mask77)
                # imsave("./test_gcn_pred/" + str(step) + "gt_mask_palm_leaf.jpg", gt_mask)

                edge_logits1 = edge_logits[0,0,:,:].cpu().numpy()
                # print(edge_logits1)


                # edge_logits2 = edge_logits[0,1,:,:].cpu().numpy()
                # fv_logits1 = fv_logits[0,:,:].cpu().numpy()
                edge_mask1 = edge_mask.cpu().numpy()[0]
                # poly_mask1 = poly_mask[0,:,:].cpu().numpy()
                # fp_mask1 = fp_mask[0,:,:].cpu().numpy()
                # img1 = data["img1"][0]
                # img2 = img2[0].cpu().numpy()
                # # print(img11.shape)
                # img11 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                # img11 = img2/255
                # img11 = cv2.resize(img1,(weight1,height1))
                # arrs = np.zeros((height1, weight1), np.uint8)
                arrs1 = np.zeros((20, 140), np.uint8)
                arrs4 = np.zeros((20, 120), np.uint8)

                # ValueError: Images of type float must be between -1 and 1.print(edge_logits.shape)
                    # print(len(pred_edge_mask))
                # for j in range(len(edge_logits1)):
                #     for k in range((len(edge_logits1[j]))):
                #         j1 = math.floor((j/8)*height1)
                #         k1 = math.floor((k/75)*weight1)
                #         # print(j,k,j1,k1)
                #         if edge_logits1[j][k]>0.65:
                #             arrs[j1][k1]= 255
                #         else:
                #             arrs[j1][k1]= 0
                for j in range(len(edge_logits1)):
                    for k in range((len(edge_logits1[j]))):
                        j1 = math.floor(j)
                        k1 = math.floor(k)
                        if edge_logits1[j][k]>0.75:
                            arrs1[j1][k1+10]= 255
                            # arrs1[j1][k1][1]= 255
                            # arrs1[j1][k1][2]= 255
                        else:
                            arrs1[j1][k1+10] = 0

                # arrs1 = cv2.medianBlur(arrs1,5)

                # for j in range(len(edge_logits1)):
                #     for k in range((len(edge_logits1[j]))):
                #         j1 = math.floor(j)
                #         k1 = math.floor(k)
                #         if arrs4[j][k]>120:
                #             arrs1[j1][k1]= 255
                #             # arrs1[j1][k1][1]= 255
                #             # arrs1[j1][k1][2]= 255
                #         else:
                #             arrs1[j1][k1] = 0


                
                # arrs4 = median
                # arrs7 = arrs1
                # arrs1 = cv2.medianBlur(arrs1,5)
                
                kernel = np.ones((3,6),np.uint8)
                # arrs1 = cv2.dilate(arrs1,kernel,iterations = 1)
                arrs1 = cv2.medianBlur(arrs1,3)
                arrs7 = arrs1
                # print(arrs1[29][0])
                
                # dilation = cv2.erode(arrs1,kernel,iterations = 1)
                dilation = cv2.dilate(arrs1,kernel,iterations = 1)
                borders = dilation ^ arrs7
                kernel1 = np.ones((1,1),np.uint8)
                # borders = cv2.erode(borders,kernel1,iterations = 1)
                dilation = np.asarray(dilation)

                # borders = cv2.medianBlur(borders,5)
                # borders = cv2.dilate(borders,kernel,iterations = 1)
                # for j in range(len(edge_logits1)):
                #     for k in range((len(edge_logits1[j]))):
                #         j1 = math.floor(j)
                #         k1 = math.floor(k)
                #         if borders[j][k]>180:
                #             borders[j1][k1]= 255
                #             # arrs1[j1][k1][1]= 255
                #             # arrs1[j1][k1][2]= 255
                #         else:
                #             borders[j1][k1] = 0



                # borders = np.zeros_like(arrs1, dtype=int)

                # # get the size of the input array in both dimensions
                # x_size, y_size = 120, 480

                # # iterate through each value
                # for i in range(x_size):
                #     for j in range(y_size):
                #         # select those which are in class 1
                #         if arrs1[i,j] == 255:
                #             # get slice of all adjacent pixels
                #             slice1 = arrs1[i-1:i+1,j-1:j+1]

                #             # check if any belong to class 2
                #             if np.any(slice1 == 0):
                #                 # update borders array
                #                 borders[i,j] = 255

                            # arrs1[j1][k1][1]= img2[j1][k1][1]
                            # arrs1[j1][k1][2]= img2[j1][k1][2]
                # im2, contours, hierarchy = cv2.findContours(arrs,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # # #print(fp_logits)
                # cont = contours[0]
                # cv2.drawContours(img11, [cont], 0, [255])
                
                # print(edge_mask.shape)
                # imsave("./test33/"+str(step)+"edge.jpg",arrs)
                # imsave("./test33/"+str(step)+"edgegt.jpg",edge_mask1)
                imsave("./test_gcn_pred/" + str(step) + "edge_leaf_borders.jpg", borders)
                # imsave("./test_gcn_pred/" + str(step) + "edgegt_leaf.jpg", edge_mask1)
                # imsave("./test33/"+str(step)+"edge1.jpg",arrs1)
                # imsave("./test33/"+str(step)+"edgegt1.jpg",edge_mask1)





                # a = data['hull'][0]
                # a = batch_ellipse[0]
                # a = np.asarray(a)


                # palm_leaf_2 = copy.deepcopy(data['img_orig'].numpy()[0])
                # cv2.polylines(palm_leaf_2, [np.int32(a)], True, [1], thickness=1)
                # for point in a:
                #     cv2.circle(palm_leaf_2, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
                # # cv2.circle(palm_leaf_2, (int(a[0][1]), int(a[0][0])), 5, (0, 255, 0), -1)

                # imsave("./test_gcn_pred/" + str(step) + "hull_on_palm_leaf.jpg", palm_leaf_2)


                # b = data['actual_gt_poly'][0]
                # palm_leaf_3 = copy.deepcopy(data['img_orig'].numpy()[0])
                # b = b.tolist()
                # cv2.polylines(palm_leaf_3, [np.int32(b)], True, [1], thickness=1)
                # for point in b:
                #     cv2.circle(palm_leaf_3, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

                # imsave("./test_gcn_pred/" + str(step) + "gt_on_palm_leaf.jpg", palm_leaf_3)













                def compute_iou_and_accuracy(arrs, edge_mask1):
                    intersection = cv2.bitwise_and(arrs, edge_mask1)
                    union = cv2.bitwise_or(arrs, edge_mask1)

                    intersection_sum = np.sum(intersection)
                    union_sum = np.sum(union)

                    iou = intersection_sum / union_sum

                    total = np.sum(arrs)
                    correct_predictions = intersection_sum

                    accuracy = correct_predictions / total
                    print(iou, accuracy)

                    return iou, accuracy
                arrs1 = np.zeros((20,120), np.float32)
                for j in range(len(edge_logits1)):
                    for k in range((len(edge_logits1[j]))):
                        j1 = math.floor(j)
                        k1 = math.floor(k)
                        if edge_logits1[j][k]>0.55:
                            arrs1[j1][k1]= 1.0
                            # arrs1[j1][k1][1]= 255
                            # arrs1[j1][k1][2]= 255
                        else:
                            arrs1[j1][k1] = 0
        #         # print("logglgog",arrs1.shape)
                # gt_mask = gt_mask.numpy()[0]
                original_mask = np.asarray(data["poly_mask"].cpu().numpy()[0])
                # print("gtgtgtgtgtgt",gt_mask.shape)
                original_mask = original_mask.astype(np.uint8)
                pred_mask = arrs1.astype(np.uint8)
                # edge_logits1 = edge_logits1.astype(np.uint8)
        #         # print(gt_mask)
        #         # pred_mask4 = pred_mask4.astype(np.uint8)



                iou1, accuracy1 = compute_iou_and_accuracy(pred_mask, original_mask)
                f = f+1
                # print("count:",f)
                # print("derived", iou1, accuracy1)

                image_url = data['image_url'][0]

                if image_url not in self.d_iou1:
                    self.d_iou1[image_url] = iou1
                    self.d_iou_c1[image_url] = 1
                else:
                    self.d_iou1[image_url] += iou1
                    self.d_iou_c1[image_url] += 1

                if image_url not in self.d_accuracy1:
                    self.d_accuracy1[image_url] = accuracy1
                    self.d_accuracy_c1[image_url] = 1
                else:
                    self.d_accuracy1[image_url] += accuracy1
                    self.d_accuracy_c1[image_url] += 1

        print('*******************aaaaaaaaaa')
        final_iou1 = dict()
        for i in self.d_iou_c1:
            final_iou1.update({i: self.d_iou1[i] / self.d_iou_c1[i]})
        iou1 = 0
        count1 = 0
        for i in final_iou1:
            iou1 += final_iou1[i]
            count1 += 1
        total_iou1 = iou1 / count1

        final_accuracy1 = dict()
        for i in self.d_accuracy_c1:
            final_accuracy1.update({i: self.d_accuracy1[i] / self.d_accuracy_c1[i]})
        accuracy1 = 0
        count1 = 0
        for i in final_accuracy1:
            accuracy1 += final_accuracy1[i]
            count1 += 1
        total_accuracy1 = accuracy1 / count1

        print("derived", total_iou1, total_accuracy1, count1)

        #         image_url = data['image_url'][0]

        #         if image_url not in self.d_iou1:
        #             self.d_iou1[image_url] = 

if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    # opts["model_path"] = './checkpoints_oldmodel_relu/epoch36_step28080.pth'
    # opts["model_path"] = './checkpoints/epoch104_step73528.pth'
    # opts ["conv_lstm_path"] = './checkpoints_oldmodel_relu11/epoch4_step15584.pth'
    tester = Tester(args, opts)
    tester.test()