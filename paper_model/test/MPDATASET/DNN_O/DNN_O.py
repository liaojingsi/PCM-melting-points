import torch
import torch.nn as nn
import torch.nn.functional as F
from paper_model.utils.pointnet2_utils import PointNetSetAbstraction_conv3_relu, PointNetSetAbstraction_relu, square_distance
from .registry import register_model

class get_model(nn.Module):
    def __init__(self, num_class_order_disorder, num_class_mol, normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction_relu(npoint=500,radius=0.5,nsample=27,in_channel=3,
                                                mlp=[8, 16, 32], group_all=False)
        self.sa2_1 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=32 + 3, mlp_kernel=[64, 96, 128], group_all=False)
        # self.sa2_2 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=64 + 3,
        #                                              mlp_kernel=[64, 64, 64], group_all=False)
        # self.sa2_3 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=64 + 3,
        #                                              mlp_kernel=[64, 64, 64], group_all=False)
        # self.sa3 = PointNetSetAbstraction_relu(None, None, None, 128 + 3, [64, 64, 64], True)
        self.sa4 = PointNetSetAbstraction_relu(None, None, None, 128 + 3, [128, 128, 128], True)
        # self.sa1 = PointNetSetAbstraction_relu(npoint=500,radius=0.5,nsample=27,in_channel=3,
        #                                         mlp=[8, 12, 16], group_all=False)
        # self.sa2_1 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=16 + 3, mlp_kernel=[16, 24, 32], group_all=False)
        # self.sa2_2 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=32 + 3, mlp_kernel=[32, 48, 64], group_all=False)
        # self.sa2_3 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=64 + 3, mlp_kernel=[64, 96, 128], group_all=False)
        # self.sa2_4 = PointNetSetAbstraction_conv3_relu(npoint=500, radius=0.5, nsample=27, in_channel=128 + 3, mlp_kernel=[128, 128, 128], group_all=False)
        # # self.sa3 = PointNetSetAbstraction_relu(None, None, None, 128 + 3, [64, 64, 64], True)
        # self.sa4 = PointNetSetAbstraction_relu(None, None, None, 128 + 3, [128, 128, 128], True)
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
        res = torch.where((dist > 0.19) & (dist < 0.21))
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

    def forward(self, input):
        B, _, _ = input.shape

        if self.normal_channel == True:
            norm1 = input[:, 0:2, :500]
            norm2 = input[:, 7:10, :500]
            temp = input[:, 6:7, :500]
            xyz = input[:, 3:6, :500]/100
            norm = torch.concat((norm1, norm2), dim=1)/100
        else:
            norm = None
            temp = input[:, 6:7, :500]
            xyz = input[:, 3:6, :500]/100

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2_1(l1_xyz, l1_points)
        # l2_xyz2, l2_points2 = self.sa2_2(l2_xyz1, l2_points1)
        # l2_xyz3, l2_points3 = self.sa2_3(l2_xyz2, l2_points2)
        # l2_xyz, l2_points = self.sa2_4(l2_xyz3, l2_points3)
        l3_xyz, l3_points = self.sa4(l2_xyz, l2_points)
        data = l2_points.permute(0,2,1)
        point = l2_xyz.permute(0,2,1)
        Tsys = torch.mean(temp, dim=2)
        correlation = self.compute_correlation(point, data)#.cuda()
        # torch.cuda.empty_cache()

        ##reg_low_temp
        sa4 = l3_points.view(B, 128)
        c = torch.cat((correlation, sa4), dim=1)

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

        return TC_pred_from_low, Tsys.squeeze(-1), l2_points,  correlation#, c, sa4, resl



@register_model
def get_cls_model(config, **kwargs):
    op_spec = config.MODEL.SPEC
    op = get_model(num_class_order_disorder=config.MODEL.NUM_CLASSES_ORDER_DISORDER,
                   num_class_mol= config.MODEL.NUM_CLASSES_MOL,
                   normal_channel=False)
    return op