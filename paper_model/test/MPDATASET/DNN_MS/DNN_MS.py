import torch
import torch.nn as nn
import torch.nn.functional as F
from paper_model.utils.pointnet2_utils import PointNetSetAbstraction_conv3_relu, PointNetSetAbstraction_relu, square_distance
from dscribe.descriptors import SOAP, MBTR
from ase import Atoms
from .registry import register_model

class get_model(nn.Module):
    def __init__(self,num_class_order_disorder, num_class_mol,normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        #soap
        self.fcsoap1 = nn.Conv2d(147, 256, 1)
        self.bnsoap1 = nn.BatchNorm2d(256)
        self.fcsoap2 = nn.Conv2d(256, 256, 1)
        self.bnsoap2 = nn.BatchNorm2d(256)
        self.fcsoap3 = nn.Conv2d(256, 256, 1)
        self.bnsoap3 = nn.BatchNorm2d(256)
        self.fcsoap4 = nn.Conv2d(256, 256, 1)
        self.bnsoap4 = nn.BatchNorm2d(256)
        self.fcsoap5 = nn.Conv2d(256, 128, 1)
        self.bnsoap5 = nn.BatchNorm2d(128)
        #mbtr
        self.fcmbtr1 = nn.Linear(157, 256)
        self.bnmbtr1 = nn.BatchNorm1d(256)
        self.fcmbtr2 = nn.Linear(256, 256)
        self.bnmbtr2 = nn.BatchNorm1d(256)
        self.fcmbtr3 = nn.Linear(256, 256)
        self.bnmbtr3 = nn.BatchNorm1d(256)
        self.fcmbtr4 = nn.Linear(256, 256)
        self.bnmbtr4 = nn.BatchNorm1d(256)
        self.fcmbtr5 = nn.Linear(256, 128)
        self.bnmbtr5 = nn.BatchNorm1d(128)
        ##reg_low_temp
        self.fcl_res0 = nn.Linear(256, 256)
        self.bnl_res0 = nn.BatchNorm1d(256)
        self.fcl_res1 = nn.Linear(256, 256)
        self.bnl_res1 = nn.BatchNorm1d(256)
        self.fcl_res2 = nn.Linear(256, 256)
        self.bnl_res2 = nn.BatchNorm1d(256)
        # self.fcl_res3 = nn.Linear(128, 128)
        # self.bnl_res3 = nn.BatchNorm1d(128)
        # self.fcl_res4 = nn.Linear(128, 128)
        # self.bnl_res4 = nn.BatchNorm1d(128)
        self.fc_lt1 = nn.Linear(256, 256)
        self.bn_lt1 = nn.BatchNorm1d(256)
        # self.fc_lt2 = nn.Linear(32, 16)
        # self.bn_lt2 = nn.BatchNorm1d(16)
        # self.fc_lt3 = nn.Linear(16, 4)
        # self.bn_lt3 = nn.BatchNorm1d(4)
        self.fc_lt4 = nn.Linear(256, 1)

        self.apply(self.weight_init)
        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def compute_correlation(self, point, data):
        #print(point.shape)
        #point = point.cpu()
        #data = data.cpu()
        b, n, _ = point.shape
        #print(data.shape)
        _, _, c = data.shape
        dist = square_distance(point, point)
        dist = torch.abs(dist)
        dist = dist ** 0.5
        #print(dist.shape)
        #print(dist)

        each_sum = torch.zeros(b, 1).cuda()
        avg_sum = torch.tensor([]).cuda()
        res = torch.where((dist > 19) & (dist < 21))
        new_psi = data.permute(0, 2, 1).unsqueeze(-1)
        corr_matrix = torch.matmul(new_psi, new_psi.permute(0, 1, 3, 2)).transpose(1, 0)
        idx = torch.stack((res[0] * n * n, res[1] * n, res[2]), dim=1)
        take_idx = torch.sum(idx, 1).unsqueeze(-1)
        num = torch.unique(res[0], return_counts=True, dim=0)
        #print('res0: ',res[0], res[0].shape)
        #print('num ', num)
        num_div = num[1].unsqueeze(-1)
        #print('num_div ', num_div)
        #print(num_div.shape)
        for c in range(c):
            take = torch.take(corr_matrix[c], take_idx).float()
            sum = each_sum.index_add(0, res[0], take)
            avg_corr = torch.div(sum, num_div)
            avg_sum = torch.cat((avg_sum, avg_corr), dim=1)

        return avg_sum

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            #print('useing weigh_init linear')
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.xavier_uniform_(m.weight)
            #print('useing weigh_init conv2d')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            #print('useing weigh_init bn2d')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            #print('useing weigh_init bn1d')

    def get_last_shared_layer(self):
        return self.sa2

    def forward(self, input):
        B, _, _ = input.shape

        temp = input[:, 6:7, :500]
        xyz = input[:, 3:6, :500]
        xyz = xyz.permute(0, 2, 1)
        # print(xyz)
        soap = input[:, 10:157, :500]
        SOAP = soap.permute(0, 2, 1)
        MBTR = input[:, :157, 500:]
        MBTR = MBTR.permute(0, 2, 1)

        SOAP = SOAP.unsqueeze(-1)
        SOAP = SOAP.permute(0, 2, 3, 1)
        local_feature = F.relu(self.bnsoap1(self.fcsoap1(SOAP)))
        local_feature = F.relu(self.bnsoap2(self.fcsoap2(local_feature)))
        local_feature = F.relu(self.bnsoap3(self.fcsoap3(local_feature)))
        local_feature = F.relu(self.bnsoap4(self.fcsoap4(local_feature)))
        local_feature = F.relu(self.bnsoap5(self.fcsoap5(local_feature)))
        local_feature = torch.max(local_feature, 2)[0]
        data = local_feature.permute(0, 2, 1)
        # point = xyz/10
        Tsys = torch.mean(temp, dim=2)
        correlation = self.compute_correlation(xyz, data)#.cuda()

        MBTR = MBTR.view(B, 157)
        global_feature = F.relu(self.bnmbtr1(self.fcmbtr1(MBTR)))
        global_feature = F.relu(self.bnmbtr2(self.fcmbtr2(global_feature)))
        global_feature = F.relu(self.bnmbtr3(self.fcmbtr3(global_feature)))
        global_feature = F.relu(self.bnmbtr4(self.fcmbtr4(global_feature)))
        global_feature = F.relu(self.bnmbtr5(self.fcmbtr5(global_feature)))

        ##reg_low_temp
        c = torch.cat((correlation, global_feature), dim=1)

        resl = F.relu(self.bnl_res0(self.fcl_res0(c)))
        resl = F.relu(self.bnl_res1(self.fcl_res1(resl)))
        resl = F.relu(self.bnl_res2(self.fcl_res2(resl)))
        # resl = F.gelu(self.bnl_res3(self.fcl_res3(resl)))
        # resl = F.gelu(self.bnl_res4(self.fcl_res4(resl)))

        to_compute_low1 = F.relu(self.bn_lt1(self.fc_lt1(resl)))
        # to_compute_low2 = F.gelu(self.bn_lt2(self.fc_lt2(to_compute_low1)))
        # to_compute_low3 = F.gelu(self.bn_lt3(self.fc_lt3(to_compute_low2)))
        to_compute_low = self.fc_lt4(to_compute_low1)
        # print('low', to_compute_low)
        # print('lv', 1/v)
        TC_pred_from_low = to_compute_low
        TC_pred_from_low = TC_pred_from_low.squeeze(-1)

        #print('temp', pred_temp)

        return TC_pred_from_low, Tsys.squeeze(-1), local_feature,  correlation#, c, sa4, resl



@register_model
def get_cls_model(config, **kwargs):
    op_spec = config.MODEL.SPEC
    op = get_model(num_class_order_disorder=config.MODEL.NUM_CLASSES_ORDER_DISORDER,
                   num_class_mol= config.MODEL.NUM_CLASSES_MOL,
                   normal_channel=False)
    return op