import os
import numpy as np
import warnings
import pickle
import copy

from paper_model.utils.pointnet2_utils import farthest_point_sample
from paper_model.utils.pointnet2_utils import pc_normalize
from paper_model.utils.pointnet2_utils import temp_normalize
from tqdm import tqdm
from torch.utils.data import Dataset

#0: 000
def expand000(points):
    points000 = copy.deepcopy(points)
    return points000

#1: 010
def expand010(points):
    points010 = np.transpose(copy.deepcopy(points), (1,0))
    points010[2] = points010[2] + 0.
    points010[3] = points010[3] + 1.
    points010[4] = points010[4] + 0.
    points010 = np.transpose(points010, (1,0))
    return points010

#2: 100
def expand100(points):
    points100 = np.transpose(copy.deepcopy(points), (1,0))
    points100[2] = points100[2] + 1.
    points100[3] = points100[3] + 0.
    points100[4] = points100[4] + 0.
    points100 = np.transpose(points100, (1,0))
    return points100

#3: 001
def expand001(points):
    points001 = np.transpose(copy.deepcopy(points), (1,0))
    points001[2] = points001[2] + 0.
    points001[3] = points001[3] + 0.
    points001[4] = points001[4] + 1.
    points001 = np.transpose(points001, (1,0))
    return points001

#4: 110
def expand110(points):
    points110 = np.transpose(copy.deepcopy(points), (1,0))
    points110[2] = points110[2] + 1.
    points110[3] = points110[3] + 1.
    points110[4] = points110[4] + 0.
    points110 = np.transpose(points110, (1,0))
    return points110

#5: 011
def expand011(points):
    points011 = np.transpose(copy.deepcopy(points), (1,0))
    points011[2] = points011[2] + 0.
    points011[3] = points011[3] + 1.
    points011[4] = points011[4] + 1.
    points011 = np.transpose(points011, (1,0))
    return points011

#6: 101
def expand101(points):
    points101 = np.transpose(copy.deepcopy(points), (1,0))
    points101[2] = points101[2] + 1.
    points101[3] = points101[3] + 0.
    points101[4] = points101[4] + 1.
    points101 = np.transpose(points101, (1,0))
    return points101

#7: 111
def expand111(points):
    points111 = np.transpose(copy.deepcopy(points), (1,0))
    points111[2] = points111[2] + 1.
    points111[3] = points111[3] + 0.
    points111[4] = points111[4] + 1.
    points111 = np.transpose(points111, (1,0))
    return points111

#8: 002
def expand002(points):
    points002 = np.transpose(copy.deepcopy(points), (1,0))
    points002[2] = points002[2] + 0.
    points002[3] = points002[3] + 0.
    points002[4] = points002[4] + 2.
    points002 = np.transpose(points002, (1,0))
    return points002

#9: 012
def expand012(points):
    points012 = np.transpose(copy.deepcopy(points), (1,0))
    points012[2] = points012[2] + 0.
    points012[3] = points012[3] + 1.
    points012[4] = points012[4] + 2.
    points012 = np.transpose(points012, (1,0))
    return points012

#10: 102
def expand102(points):
    points102 = np.transpose(copy.deepcopy(points), (1,0))
    points102[2] = points102[2] + 1.
    points102[3] = points102[3] + 0.
    points102[4] = points102[4] + 2.
    points102 = np.transpose(points102, (1,0))
    return points102

#11: 112
def expand112(points):
    points112 = np.transpose(copy.deepcopy(points), (1,0))
    points112[2] = points112[2] + 1.
    points112[3] = points112[3] + 1.
    points112[4] = points112[4] + 2.
    points112 = np.transpose(points112, (1,0))
    return points112

#12: 200
def expand200(points):
    points200 = np.transpose(copy.deepcopy(points), (1,0))
    points200[2] = points200[2] + 2.
    points200[3] = points200[3] + 0.
    points200[4] = points200[4] + 0.
    points200 = np.transpose(points200, (1,0))
    return points200

#13: 210
def expand210(points):
    points210 = np.transpose(copy.deepcopy(points), (1,0))
    points210[2] = points210[2] + 2.
    points210[3] = points210[3] + 1.
    points210[4] = points210[4] + 0.
    points210 = np.transpose(points210, (1,0))
    return points210

#14: 201
def expand201(points):
    points201 = np.transpose(copy.deepcopy(points), (1,0))
    points201[2] = points201[2] + 2.
    points201[3] = points201[3] + 0.
    points201[4] = points201[4] + 1.
    points201 = np.transpose(points201, (1,0))
    return points201

#15: 211
def expand211(points):
    points211 = np.transpose(copy.deepcopy(points), (1,0))
    points211[2] = points211[2] + 2.
    points211[3] = points211[3] + 1.
    points211[4] = points211[4] + 1.
    points211 = np.transpose(points211, (1,0))
    return points211

#16: 020
def expand020(points):
    points020 = np.transpose(copy.deepcopy(points), (1,0))
    points020[2] = points020[2] + 0.
    points020[3] = points020[3] + 2.
    points020[4] = points020[4] + 0.
    points020 = np.transpose(points020, (1,0))
    return points020

#17: 120
def expand120(points):
    points120 = np.transpose(copy.deepcopy(points), (1,0))
    points120[2] = points120[2] + 1.
    points120[3] = points120[3] + 2.
    points120[4] = points120[4] + 0.
    points120 = np.transpose(points120, (1,0))
    return points120

#18: 021
def expand021(points):
    points021 = np.transpose(copy.deepcopy(points), (1,0))
    points021[2] = points021[2] + 0.
    points021[3] = points021[3] + 2.
    points021[4] = points021[4] + 1.
    points021 = np.transpose(points021, (1,0))
    return points021

#19: 121
def expand121(points):
    points121 = np.transpose(copy.deepcopy(points), (1,0))
    points121[2] = points121[2] + 1.
    points121[3] = points121[3] + 2.
    points121[4] = points121[4] + 1.
    points121 = np.transpose(points121, (1,0))
    return points121

#20: 220
def expand220(points):
    points220 = np.transpose(copy.deepcopy(points), (1,0))
    points220[2] = points220[2] + 2.
    points220[3] = points220[3] + 2.
    points220[4] = points220[4] + 0.
    points220 = np.transpose(points220, (1,0))
    return points220

#21: 221
def expand221(points):
    points221 = np.transpose(copy.deepcopy(points), (1,0))
    points221[2] = points221[2] + 2.
    points221[3] = points221[3] + 2.
    points221[4] = points221[4] + 1.
    points221 = np.transpose(points221, (1,0))
    return points221

#22: 202
def expand202(points):
    points202 = np.transpose(copy.deepcopy(points), (1,0))
    points202[2] = points202[2] + 2.
    points202[3] = points202[3] + 0.
    points202[4] = points202[4] + 2.
    points202 = np.transpose(points202, (1,0))
    return points202

#23: 212
def expand212(points):
    points212 = np.transpose(copy.deepcopy(points), (1,0))
    points212[2] = points212[2] + 2.
    points212[3] = points212[3] + 1.
    points212[4] = points212[4] + 2.
    points212 = np.transpose(points212, (1,0))
    return points212

#24: 022
def expand022(points):
    points022 = np.transpose(copy.deepcopy(points), (1,0))
    points022[2] = points022[2] + 0.
    points022[3] = points022[3] + 2.
    points022[4] = points022[4] + 2.
    points022 = np.transpose(points022, (1,0))
    return points022

#25: 122
def expand122(points):
    points122 = np.transpose(copy.deepcopy(points), (1,0))
    points122[2] = points122[2] + 1.
    points122[3] = points122[3] + 2.
    points122[4] = points122[4] + 2.
    points122 = np.transpose(points122, (1,0))
    return points122

#26: 222
def expand222(points):
    points222 = np.transpose(copy.deepcopy(points), (1,0))
    points222[2] = points222[2] + 2.
    points222[3] = points222[3] + 2.
    points222[4] = points222[4] + 2.
    points222 = np.transpose(points222, (1,0))
    return points222


def expand(points):
    points000 = expand000(points)
    points010 = expand010(points)
    points100 = expand100(points)
    points001 = expand001(points)
    points110 = expand110(points)
    points011 = expand011(points)
    points101 = expand101(points)
    points111 = expand111(points)
    points002 = expand002(points)
    points012 = expand012(points)
    points102 = expand102(points)
    points112 = expand112(points)
    points200 = expand200(points)
    points210 = expand210(points)
    points201 = expand201(points)
    points211 = expand211(points)
    points020 = expand020(points)
    points120 = expand120(points)
    points021 = expand021(points)
    points121 = expand121(points)
    points220 = expand220(points)
    points221 = expand221(points)
    points202 = expand202(points)
    points212 = expand212(points)
    points022 = expand022(points)
    points122 = expand122(points)
    points222 = expand222(points)
    expand_points = np.concatenate((points000,
                                    points010,
                                    points100,
                                    points001,
                                    points110,
                                    points011,
                                    points101,
                                    points111,
                                    points002,
                                    points012,
                                    points102,
                                    points112,
                                    points200,
                                    points210,
                                    points201,
                                    points211,
                                    points020,
                                    points120,
                                    points021,
                                    points121,
                                    points220,
                                    points221,
                                    points202,
                                    points212,
                                    points022,
                                    points122,
                                    points222), axis=0)
    return expand_points

class multi_DataLoader_reg(Dataset):
    def __init__(self,
                 cfg,
                 split='train'):
        self.root = cfg.DATASET.ROOT
        print(self.root)
        self.npoints = cfg.DATASET.NUM_POINT
        self.process_data = cfg.DATASET.PROCESS_DATA
        self.uniform = cfg.DATASET.USE_UNIFORM_SAMPLE
        self.use_normals = cfg.DATASET.USE_NORMALS
        self.catfile1 = os.path.join(self.root, 'crystal_id_list.txt')
        print(self.catfile1)
        self.catfile2 = os.path.join(self.root, 'phase_id.txt')
        #self.catfile3 = os.path.join(self.root, 'tc_list.txt')

        self.cat1 = [line.rstrip() for line in open(self.catfile1)]
        self.classes1 = dict(zip(self.cat1, range(len(self.cat1))))

        self.cat2 = [line.rstrip() for line in open(self.catfile2)]
        self.classes2 = dict(zip(self.cat2, range(len(self.cat2))))

        #self.cat3 = [line.rstrip() for line in open(self.catfile3)]

        crystal_ids = {}

        crystal_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'crystal_train.txt'))]
        crystal_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'crystal_test.txt'))]

        assert (split == 'train' or split == 'test')
        crystal_names = ['_'.join(x.split('_')[0:-4]) for x in crystal_ids[split]]
        #print(crystal_names)
        phase_names = ['_'.join(x.split('_')[1:-3]) for x in crystal_ids[split]]
        melting_points = ['_'.join(x.split('_')[2:-2]) for x in crystal_ids[split]]
        self.datapath = [(crystal_names[i], phase_names[i], melting_points[i], os.path.join(self.root, crystal_names[i], crystal_ids[split][i]) + '.txt')
                         for i in range(len(crystal_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(cfg.DATASET.ROOT, 'net_%s_%dpts_fps.dat' % (split, self.npoints))
        else:
            self.save_path = os.path.join(cfg.DATASET.ROOT, 'net_%s_%dpts.dat' % (split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels1 = [None] * len(self.datapath)
                self.list_of_labels2 = [None] * len(self.datapath)
                self.list_of_temp = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    #print("print", self.datapath[index][0])
                    cls1 = self.classes1[self.datapath[index][0]]
                    cls1 = np.array([cls1]).astype(np.int32)
                    cls2 = self.classes2[self.datapath[index][1]]
                    cls2 = np.array([cls2]).astype(np.int32)
                    reg = self.datapath[index][2]
                    reg = np.array([reg]).astype(np.float32)
                    point_set = np.loadtxt(fn[3], delimiter=',').astype(np.float32)
                    #point_set = expand(origin_point_set)
                    #print(point_set[997:1001])

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    # else:
                    #     point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels1[index] = cls1
                    self.list_of_labels2[index] = cls2
                    self.list_of_temp[index] = reg

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels1, self.list_of_labels2, self.list_of_temp], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels1, self.list_of_labels2, self.list_of_temp = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label1, label2, temp_label = self.list_of_points[index], self.list_of_labels1[index], self.list_of_labels2[index], self.list_of_temp[index]
            if point_set.shape[0] != 501:
                print(point_set.shape, self.datapath[index][3])
        else:
            fn = self.datapath[index]
            cls1 = self.classes1[self.datapath[index][0]]
            cls2 = self.classes2[self.datapath[index][1]]
            reg = self.datapath[index][2]
            label1 = np.array([cls1]).astype(np.int32)
            label2 = np.array([cls2]).astype(np.int32)
            temp_label = np.array([reg]).astype(np.float32)
            point_set = np.loadtxt(fn[3], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            # else:
            #     point_set = point_set[0:self.npoints, :]

            if point_set.shape[0] != 501:
                print(point_set.shape, fn)

        point_set[:500, 3:6] = pc_normalize(point_set[:500, 3:6])
        # temp_label = temp_normalize(temp_label)
        if not self.use_normals:
            point_set[:500, 3:6] = point_set[:500, 3:6]
        # if self.use_normals:
        #     point_set = point_set[:, 3:6]
        #     temp_label = temp_normalize(temp_label)
        # else:
        #     point_set = point_set[:, 3:6]


        # if label1[0] == 0:
        #     temp = 361.0
        # elif label1[0] == 1:
        #     temp = 361.0
        # elif label1[0] == 2:
        #     temp = 439.0
        # elif label1[0] == 3:
        #     temp = 401.0
        # elif label1[0] == 4:
        #     temp = 419.0
        # elif label1[0] == 5:
        #     temp = 400.0
        # elif label1[0] == 6:
        #     temp = 374.0
        # elif label1[0] == 7:
        #     temp = 366.0
        # elif label1[0] == 8:
        #     temp = 376.0
        # elif label1[0] == 9:
        #     temp = 431.0
        # elif label1[0] == 10:
        #     temp = 383.0
        # elif label1[0] == 11:
        #     temp = 429.0
        #print(point_set, label1[0], label2[0], temp_label[0])

        return point_set, label1[0], label2[0], temp_label[0]

    def __getitem__(self, index):
        return self._get_item(index)
