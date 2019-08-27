import torch
import json
import os
import argparse

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict

import edge_imageprovider as image_provider
from tqdm import tqdm
from losses import poly_mathcing_loss
import cv2
import numpy as np

from edgeonly import Model
from skimage.io import imsave
import matplotlib.pyplot as plt




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")

def clockwise_check(points):
    sum = 0
    for i in range(len(points)):
        if i != len(points) - 1:
            sum_x = points[i+1][0] - points[i][0]
            sum_y = points[i+1][1] + points[i][1]
            sum += sum_x * sum_y
        else:
            sum_x = points[0][0] - points[i][0]
            sum_y = points[0][1] + points[i][1]
            sum += sum_x * sum_y
    if sum > 0:
        return True
    else:
        return False

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




def create_folder(path):
    # if os.path.exists(path):
    # resp = input 'Path %s exists. Continue? [y/n]'%path
    #    if resp == 'n' or resp == 'N':
    #       raise RuntimeError()

    # else:
    os.system('mkdir -p %s' % path)
    print('Experiment folder created at: %s' % path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args




def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    #done

    train_split = 'train'
    train_val_split ='train_val'

    dataset_train = DataProvider(split=train_split, opts=opts[train_split], mode=train_split)
    dataset_val = DataProvider(split=train_val_split, opts=opts[train_val_split], mode=train_val_split)
    # weight = dataset_train.getweight()
    # label_to_count = dataset_train.getlabeltocount()
    # label_to_countval = dataset_val.getlabeltocount()
    train_loader = DataLoader(dataset_train, batch_size=opts[train_split]['batch_size'],
                              shuffle=True, num_workers=opts[train_split]['num_workers'],
                              collate_fn=image_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts[train_val_split]['batch_size'],
                            shuffle=False, num_workers=opts[train_val_split]['num_workers'],
                            collate_fn=image_provider.collate_fn)

    return train_loader, val_loader


class Trainer(object):
    def __init__(self, args, opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts

        self.model_path = './checkpoints_cgcn/epoch10_step7610.pth'

        self.model = Model(160,960,3).to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])

        create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints_cgcn44'))

        # Copy experiment file
        os.system('cp %s %s' % (args.exp, self.opts['exp_dir']))



        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)




        self.weight = torch.tensor([10.0, 2.0, 6.0, 20.0, 20.0, 20.0]).to(device)
        self.p_n = torch.tensor([0.65]).to(device)
        self.p_n1 = torch.tensor([0.75]).to(device)
        self.edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.p_n)
        self.edge_loss_fn1 = nn.BCELoss(reduction = 'none')
        self.p_n2 = torch.tensor([80.0]).to(device)



        self.fp_loss_fn = nn.MSELoss()
        self.gcn_loss_sum_train = 0
        self.gcn_loss_sum_val = 0
        self.x1 = []
        self.y1 = []
        self.y2 = []


        # OPTIMIZER
        no_wd = []
        wd = []
        print('Weight Decay applied to: ')

        # ct = 0
        # for child in self.model.children():
        #     ct += 1
        #     # print(child)
        #     if ct>=8 and ct<=9:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #             print(child)



        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                print(name,)



        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'],
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'],
                                                  gamma=0.1)
        # if args.resume is not None:
        #     self.resume(args.resume)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'gcn_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints_cgcn', 'epoch%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        # self.conv_lstm.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])


        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s' % (
        self.epoch, self.global_step, path))

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            if self.epoch % 5 == 0 and self.epoch > 0:
                print("saved222222222222222222333333333333333333333333333333")
                self.save_checkpoint(epoch)
            if self.epoch >= 140:
                break
            if epoch >= 1:
                self.x1.append(epoch)
                self.y1.append(self.gcn_loss_sum_train)
                self.y2.append(self.gcn_loss_sum_val)
                if self.epoch >= 4:
                    plt.plot(self.x1, self.y1, label = "Training loss")
                    plt.plot(self.x1, self.y2, label = "Validation loss")
                    plt.xlabel('epochs')
                    plt.ylabel('Loss')
                    plt.title('GCN Loss')
                    plt.legend()
                    plt.savefig('foo.png') 


            #            self.lr_decay.step()
            #            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)

    def train(self, epoch):
        print('Starting training')
        self.model.train()
        losses = []
        gcn_losses = []
        edge_losses = []
        vertex_losses = []
        accum = defaultdict(float)

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()

            img = data['img']
            img = torch.cat(img)

            img = img.view(-1, 160, 960, 3)
            img = torch.transpose(img, 1, 3)
            img = torch.transpose(img, 2, 3)
            img = img.float()

            # poly_mask = data["poly_mask"]

            bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
            hull = torch.from_numpy(np.asarray(data['hull'])).to(device)
            # hull = []
            gt_mask = torch.from_numpy(np.asarray(data['poly_mask'])).to(device)

            # hull = []



            # edge_logits, tg2 = self.model(img.to(device), bbox,hull, gt_mask)
            edge_logits, class_prob = self.model(img.to(device))




            edge_mask = data["edge_mask"]
            poly_mask = data["poly_mask"]
            vertex_mask = data["vertices_mask"]

            gt_label = data["gt_label"]

            self.optimizer.zero_grad()



            # weight = torch.tensor([10.0, 2.0, 6.0, 20.0, 20.0, 20.0, 20.0])

            # poly_loss = self.edge_loss_fn1(poly_logits, poly_mask.to(device))
            edge_loss = self.edge_loss_fn(edge_logits[:,0,:,:], poly_mask.to(device))
            
            class_loss1 = self.edge_loss_fn1(class_prob, gt_label.to(device))

            pt = torch.exp(-class_loss1)
            class_loss = self.weight * (1-pt)**2 * class_loss1

            # class_loss = class_loss * self.weight
            class_loss = class_loss.mean()

            # vertex_loss = self.edge_loss_fn1(edge_logits[:,1,:,:], poly_mask.to(device))








            #####################################################################################################



            # loss_sum = 0
            # self.gcn_loss_sum_train = 0
            # edge_loss_sum = 0

            # w1 = torch.tensor(data["w"])
            # h1 = torch.tensor(data["h"])

            # pred_cps = output_dict['pred_polys'][-1]
            # # print(pred_cps.shape)
            # # print("x1:",pred_cps[0,0,0],pred_cps[0,10,0],pred_cps[0,15,0])
            # # print("y1:",pred_cps[0,0,1],pred_cps[0,10,1],pred_cps[0,15,1])
            # # pred_cps_x = pred_cps[:, :, 1].view(-1, 350, 1)
            # # # pred_cps_x = pred_cps_x/float(h1[0])
            # # pred_cps_y = pred_cps[:, :, 0].view(-1, 350, 1)
            # # pred_cps_y = pred_cps_y/float(w1[0])

            # # pred_cps = torch.cat((pred_cps_x, pred_cps_y), dim=2)
            # #print("predpred",pred_cps.shape)

            # # pred_polys = self.spline.sample_point(pred_cps)

            # dp = data['actual_gt_poly']

            # # for i in range(len(dp)):
            # #     if clockwise_check(dp[i]) == False:
            # #         idx = [i for i in range(dp[i].size(0)-1, -1, -1)]
            # #         idx = torch.LongTensor(idx)
            # #         dp[i] = dp[i].index_select(0, idx)

            # dp = uniformsample_batch(dp, 1600)

            # dp = torch.stack(dp)
            # #print("dpdpdpddp",dp.shape)


            # dp_x = dp[:, :, 1].view(-1, 1600, 1)
            # # dp_x = dp_x/float(h1[0])
            # dp_y = dp[:, :, 0].view(-1, 1600, 1)
            # # dp_y = dp_y/float(w1[0])
            # dp = torch.cat((dp_x, dp_y), dim=2)

            # dp = torch.tensor(dp)
            # # print(dp.shape)
            # # print("x2:",dp[0,0,0],dp[0,10,0],dp[0,15,0])
            # # print("y2:",dp[0,0,1],dp[0,10,1],dp[0,15,1])
            # # print("dpdpddpdpdpdpdpddpdddddddppddpdpdpdpdpd", dp.shape)

            # gt_right_order, poly_mathcing_loss_sum = poly_mathcing_loss(self.opts['p_num'],
            #                                                             pred_cps,
            #                                                             dp,
            #                                                             loss_type=self.opts['loss_type'])
            poly_mathcing_loss_sum = 0
            loss_v = 70*class_loss + 50*edge_loss
            loss_sum = edge_loss + class_loss
            self.gcn_loss_sum_train = class_loss
            edge_loss_sum = edge_loss
            vertex_loss_sum = 0


            loss_v.backward()
            # self.y1.append(self.gcn_loss_sum_train)

            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.gcn_model.parameters(), self.opts['grad_clip'])

            self.optimizer.step()


            loss = loss_sum.item()

            losses.append(loss)
            gcn_losses.append(self.gcn_loss_sum_train)
            edge_losses.append(edge_loss_sum.item())
            vertex_losses.append(0)



            accum['loss'] += float(loss)
            accum['gcn_loss'] += float(self.gcn_loss_sum_train)
            accum['edge_loss'] += float(edge_loss_sum)
            accum['vertex_loss'] += float(vertex_loss_sum)
            accum['length'] += 1

            if (step % self.opts['print_freq'] == 0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f, GCN Loss: %f, Edge Loss: %f,  Vertex Loss: %f" % (
                str(datetime.now()), epoch, self.global_step, accum['loss'], accum['gcn_loss'], accum['edge_loss'], accum['vertex_loss']))
                accum = defaultdict(float)

            self.global_step += 1
        avg_epoch_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]

        avg_epoch_loss = avg_epoch_loss / len(losses)
        self.gcn_loss_sum_train = avg_epoch_loss

        print("Average Epoch %d loss is : %f" % (epoch, avg_epoch_loss))

    def validate(self):
        print('Validating')
        self.model.eval()
        losses = []
        gcn_losses = []
        edge_losses = []
        vertex_losses = []
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):

                img = data['img']
                img = torch.cat(img)

                img = img.view(-1, 160, 960, 3)
                img = torch.transpose(img, 1, 3)
                img = torch.transpose(img, 2, 3)
                img = img.float()



                bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
                hull = torch.from_numpy(np.asarray(data['hull'])).to(device)
                # hull = []
                gt_mask = torch.from_numpy(np.asarray(data['poly_mask'])).to(device)
                # hull = []
                # print(hull.shape)



                

                # edge_logits = self.model(img.to(device), bbox, hull, gt_mask)
                edge_logits, class_prob = self.model(img.to(device))


                edge_mask = data["edge_mask"]
                poly_mask = data["poly_mask"]
                vertex_mask = data["vertices_mask"]

                gt_label = data["gt_label"]

                self.optimizer.zero_grad()

                # weight = torch.tensor([10.0, 2.0, 6.0, 20.0, 20.0, 20.0, 20.0])

                # poly_loss = self.edge_loss_fn1(poly_logits, poly_mask.to(device))
                edge_loss = self.edge_loss_fn(edge_logits[:,0,:,:], poly_mask.to(device))

                class_loss1 = self.edge_loss_fn1(class_prob, gt_label.to(device))

                pt = torch.exp(-class_loss1)
                class_loss = self.weight * (1-pt)**2 * class_loss1

            # class_loss = class_loss * self.weight
                class_loss = class_loss.mean()
                # class_loss = class_loss * self.weight
                # class_loss = class_loss.mean()
                # vertex_loss = self.edge_loss_fn1(edge_logits[:,1,:,:], poly_mask.to(device))

                #####################################################################################################

                # loss_sum = 0
                # self.gcn_loss_sum_val = 0
                # edge_loss_sum = 0

                # pred_cps = output_dict['pred_polys'][-1]
                # # print("pred_shape",pred_cps.shape)
                # # print(pred_cps.shape)
                # # print("size_cps",max(pred_cps[0,:,0]),max(pred_cps[0,:,1]))
                # # print("x1:",pred_cps[0,0,0],pred_cps[0,10,0],pred_cps[0,15,0], pred_cps[0,30,0])
                # # print("y1:",pred_cps[0,0,1],pred_cps[0,10,1],pred_cps[0,15,1],pred_cps[0,30,1])

                # # pred_polys = self.spline.sample_point(pred_cps)

                # dp = data['actual_gt_poly']
                # # for i in range(len(dp)):
                # #     if clockwise_check(dp[i]) == False:
                # #         idx = [i for i in range(dp[i].size(0)-1, -1, -1)]
                # #         idx = torch.LongTensor(idx)
                # #         dp[i] = dp[i].index_select(0, idx)
                # dp = uniformsample_batch(dp, 1600)

                # dp = torch.stack(dp)

                # dp_x = dp[:, :, 1].view(-1, 1600, 1)
                # dp_y = dp[:, :, 0].view(-1, 1600, 1)
                # # dp8 = torch.cat((dp_y, dp_x), dim=2) 
                # dp = torch.cat((dp_x, dp_y), dim=2)

                # # print("dp_shape",dp.shape)
                # # print("size_gt:",max(dp[0,:,0]),max(dp[0,:,1]))
                # # print("x2:",dp[0,0,0],dp[0,10,0],dp[0,15,0], dp[0,30,0])
                # # print("y2:",dp[0,0,1],dp[0,10,1],dp[0,15,1], dp[0,30,1])

                # # print("dpdpddpdpdpdpdpddpdddddddppddpdpdpdpdpd", pred_cps.shape)


                # # dp2 = np.floor(pred_cps[0].cpu().numpy())
                # # dp21 = dp2[:,0]
                # # dp22 = dp2[:,1]
                # # dp2 = [[a] + [b] for a, b in zip(dp22, dp21)]
                # # print("ssssssssssssss",dp2[100],dp2[200])
                # # arrs4 = data["img_orig"][0]

                # # cv2.polylines(arrs4, np.int32([dp2]), True, [1], thickness=1)

                # # imsave("./test_gcn_pred/" + str(step) + "pred_on_palm_leaf.jpg", arrs4)

                # gt_right_order, poly_mathcing_loss_sum = poly_mathcing_loss(self.opts['p_num'],
                #                                                             pred_cps,
                #                                                             dp,
                #                                                             loss_type=self.opts['loss_type'])
                poly_mathcing_loss_sum = class_loss
                loss_sum = 70*class_loss + 50*edge_loss
                self.gcn_loss_sum_val = class_loss
                edge_loss_sum = edge_loss
                # vertex_loss_sum = vertex_loss



                loss = loss_sum.item()
                losses.append(loss)
                gcn_losses.append(self.gcn_loss_sum_val)
                edge_losses.append(edge_loss_sum.item())
                vertex_losses.append(0)
                # fp_losses.append(fp_loss.item())

        avg_epoch_loss = 0.0
        avg_gcn_loss = 0.0
        avg_edge_loss = 0.0
        avg_vertex_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_gcn_loss += gcn_losses[i]
            avg_edge_loss += edge_losses[i]
            avg_vertex_loss += vertex_losses[i]


        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_gcn_loss = avg_gcn_loss / len(losses)
        avg_edge_loss = avg_edge_loss / len(losses)
        avg_vertex_loss = avg_vertex_loss / len(losses)
        self.gcn_loss_sum_val = avg_gcn_loss

        print("Average VAL error is : %f, Average VAL gcn error is : %f, Average VAL edge error is : %f, Average VAL vertex error is : %f" % (avg_epoch_loss, avg_gcn_loss, avg_edge_loss, avg_vertex_loss))
        self.model.train()


if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args, opts)
    trainer.loop()
