import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_height, input_width, input_channels):
        super(Model, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.layer11 = nn.Sequential(nn.Conv2d(input_channels, 8, 3, padding=1, bias=True), nn.BatchNorm2d(8),
                                     nn.ReLU(), nn.MaxPool2d(2))
        self.layer12 = nn.Sequential(nn.Conv2d(input_channels, 8, 5, padding=2, bias=True), nn.BatchNorm2d(8),
                                     nn.ReLU(), nn.MaxPool2d(2))



        self.layer_d1 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=True), nn.BatchNorm2d(32),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True))
        self.layer_d1s = nn.Conv2d(16, 32, 1, stride=2)
        self.relu_d1 = nn.ReLU()



        self.layer_d2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True), nn.BatchNorm2d(64),
                                      nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True))
        self.layer_d2s = nn.Conv2d(32, 64, 1, stride=1)
        self.relu_d2 = nn.ReLU()


        #
        # self.layer6 = nn.Sequential(nn.Conv2d(64, 8, 3, padding=1, stride=2, bias=True), nn.ReLU())
        # self.layer7 = nn.Sequential(nn.Linear(12000, 1500, bias=True), nn.ReLU())




        self.layer_d3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=True), nn.BatchNorm2d(128),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1, stride=1, bias=True))
        self.layer_d3s = nn.Conv2d(64, 128, 1, stride=2)
        self.relu_d3 = nn.ReLU()




        self.layer8 = nn.Sequential(nn.Conv2d(128, 10, 1, padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer9 = nn.Sequential(nn.Linear(6000, 2400, bias=True))

        self.layer88 = nn.Sequential(nn.Conv2d(128, 200, (3,5), padding=(1,2), stride = (1,2), bias=True), nn.ReLU())
        self.layer89 = nn.Sequential(nn.Conv2d(200, 256, 1, padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer90 = nn.Sequential(nn.Conv2d(256, 100, 1, padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer14 = nn.Sequential(nn.Linear(7500, 2000, bias=True), nn.ReLU(), nn.Dropout(p = 0.3))
        self.layer15 = nn.Sequential(nn.Linear(2000, 500, bias=True), nn.ReLU(), nn.Dropout(p = 0.3))
        self.layer16 = nn.Sequential(nn.Linear(500, 6, bias=True), nn.Softmax())


        # self.layer40 = nn.Sequential(nn.Conv2d(128, 8, 3, padding=1, bias=True), nn.ReLU())
        # self.layer41 = nn.Sequential(nn.Linear(12000, 1500, bias=True), nn.ReLU())

    def forward(self, input_img):
        output11 = self.layer11(input_img)
        output12 = self.layer12(input_img)
        output = torch.cat((output11, output12), 1)


        output_d1 = self.layer_d1(output)
        output_d1s = self.layer_d1s(output)
        output = output_d1 + output_d1s
        output = self.relu_d1(output)



        output_d2 = self.layer_d2(output)
        output_d2s = self.layer_d2s(output)
        output = output_d2 + output_d2s
        output = self.relu_d2(output)

        # poly_logits = self.layer6(output)
        # poly_feats = poly_logits
        # poly_logits = poly_logits.view(poly_logits.shape[0], -1)
        # poly_logits = self.layer7(poly_logits)
        # poly_logits44 = poly_logits.view(poly_logits.shape[0], 1, 25, 60)
        #
        # poly_logits1 = poly_logits.view(poly_logits.shape[0], 25, 60)



        output_d3 = self.layer_d3(output)
        output_d3s = self.layer_d3s(output)
        output = output_d3 + output_d3s
        output = self.relu_d3(output)




        # print(output.shape)
        # print(output.shape)
        edge_logits = self.layer8(output)
        edge_feats = edge_logits
        # edge_feats1 = edge_logits
        edge_logits = edge_logits.view(edge_logits.shape[0], -1)
        edge_logits = self.layer9(edge_logits)
        edge_logits44 = edge_logits.view(edge_logits.shape[0], 1, 20, 120)

        edge_feats1 = self.layer88(output)
        edge_feats1 = self.layer89(edge_feats1)
        edge_feats1 = self.layer90(edge_feats1)
        edge_feats1 = edge_feats1.view(edge_logits.shape[0], -1)
        class_prob = self.layer14(edge_feats1)
        class_prob = self.layer15(class_prob)
        class_prob2 = self.layer16(class_prob)
        # class_prob2 = F.log_softmax(class_prob)


        # vertex_logits = self.layer40(output)
        # # edge_feats = edge_logits
        # vertex_logits = vertex_logits.view(vertex_logits.shape[0], -1)
        # vertex_logits = self.layer41(vertex_logits)
        # vertex_logits44 = vertex_logits.view(vertex_logits.shape[0], 1, 25, 60)

        # tg2 = output
        # edge_feats = nn.Upsample(mode='bilinear', scale_factor=2)(edge_feats)
        # print(edge_feats.shape)
        # tg2 = torch.cat((output,edge_feats,edge_logits44),1)
        # tg2 = tg2.view(edge_logits.shape[0],1,25,60)

        edge_logits1 = edge_logits.view(edge_logits.shape[0],1, 20, 120)
        # vertex_logits1 = vertex_logits.view(edge_logits.shape[0], 25, 60)


        return edge_logits1, class_prob2
