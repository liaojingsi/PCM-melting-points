import os
import pandas as pd
import numpy as np
from dscribe.descriptors import SOAP, MBTR
from ase import Atoms

id = 1148418
n_atom = 13
melt = 48+273
crytempstart = 280
liqtempstart = 340

def dst(xyz1, xyz2):
    dst_2x1x2 = -2 * np.matmul(xyz1, xyz2.T)
    # print(dst_2x1x2.shape)
    dst_x12 = np.sum(xyz1 ** 2, axis=1).reshape(xyz1.shape[0], 1)
    dst_x12 = np.repeat(dst_x12, repeats=xyz1.shape[0], axis=1)
    # print(dst_x12)
    dst_x22 = np.sum(xyz2.T ** 2, axis=0).reshape(1, xyz1.shape[0])
    dst_x22 = np.repeat(dst_x22, repeats=xyz2.shape[0], axis=0)
    # print(dst_x22)
    dst = dst_x12 + dst_2x1x2 + dst_x22
    # print(dst)
    return dst

def read_dump(file, num_atom, id, order, melt, temper):
    data_to_get = file
    data_to_write = []
    cell_data = pd.read_csv(data_to_get, sep=' ', names=['id', 'mol',
                                                    'atom', 'mass', 'q',
                                                    'vx', 'vy', 'vz',
                                                    'ke', 'fx', 'fy', 'fz',
                                                    'pe1', 'pe2',
                                                    'stress11', 'stress12', 'stress13',
                                                    'stress14', 'stress15', 'stress16',
                                                    'stress21', 'stress22', 'stress23',
                                                    'stress24', 'stress25', 'stress26',
                                                    'x', 'y', 'z',
                                                    'xs', 'ys', 'zs', 'useless1', 'useless2'], low_memory=False)
    cell_data = cell_data.applymap(lambda x: x.replace('+', '') if isinstance(x, str) else x)
    # cell_data = cell_data.applymap(lambda x: x.replace('-', '') if isinstance(x, str) else x)
    data = pd.read_csv(data_to_get, sep=' ', names=['id', 'mol',
                                                    'atom', 'mass', 'q',
                                                    'vx', 'vy', 'vz',
                                                    'ke', 'fx', 'fy', 'fz',
                                                    'pe1', 'pe2',
                                                    'stress11', 'stress12', 'stress13',
                                                    'stress14', 'stress15', 'stress16',
                                                    'stress21', 'stress22', 'stress23',
                                                    'stress24', 'stress25', 'stress26',
                                                    'x', 'y', 'z',
                                                    'xs', 'ys', 'zs', 'useless1', 'useless2'], low_memory=False)
    data = data.drop(labels=['useless1', 'useless2'], axis=1)
    data = data.dropna(axis=0, how='any')
    #data = data[['id', 'atom', 'mass', 'q', 'vx', 'vy', 'vz', 'xs', 'ys', 'zs']]
    data = data.reset_index(drop=True)
    # print(data)
    count = 0
    row = 0
    for i in range(data.shape[0]):  # (计算数据帧的行数数量，每个数据帧有2个‘ITEM:’)
        row = row + 1
        if data['id'].iloc[i] == 'ITEM:':
            count = count + 1
        if count == 2:  # 数到第二张数据帧的第一个‘ITEM:’
            break
    num_elements_with_tile = row - 1
    # print(num_elements_with_tile)
    need_drop = data.shape[0] % num_elements_with_tile
    if need_drop == 0:  # 判断数据帧是否完整，计算完整数据帧的数量
        num_dataframes = int(data.shape[0] / num_elements_with_tile)
    else:
        num_dataframes = int((data.shape[0] - need_drop) / num_elements_with_tile)
        data = data.drop(data.tail(need_drop).index)
    for j in range(num_dataframes):  # 分拆数据帧表，删除每一帧不需要的的前两列和前两行，并重新编号
        data_to_write = data.iloc[j * num_elements_with_tile: (j + 1) * num_elements_with_tile]
        data_to_write = data_to_write.reset_index(drop=True)
        data_to_write = data_to_write.drop(labels=[0], axis=0)  # 删前一行
        #locals()[f'data_{j}'] = locals()[f'data_{j}'].drop(labels=['atom'], axis=1)  # 删列
        data_to_write = data_to_write.drop(labels=[ 'atom', 'mass', 'q',
                                                    'vx', 'vy', 'vz',
                                                    'fx', 'fy', 'fz',
                                                    'pe1', 'pe2',
                                                    'stress11', 'stress12', 'stress13',
                                                    'stress14', 'stress15', 'stress16',
                                                    'stress21', 'stress22', 'stress23',
                                                    'stress24', 'stress25', 'stress26',
                                                    'xs', 'ys', 'zs'], axis=1)
        data_to_write = data_to_write.reset_index(drop=True)  # 重编号
        data_to_write = pd.DataFrame(data_to_write, dtype=float)
        data_to_write = data_to_write.sort_values(by=['mol', 'id'])
        data_to_write = data_to_write.reset_index(drop=True)
        data_to_write = data_to_write.values
        avg = np.mean(data_to_write[:, 2], axis=0)
        temp = 335.905 * avg
        frame_data = data_to_write.reshape((-1, num_atom, 6))
        # print(frame_data.shape)
        directions = []
        for frame in frame_data:
            # print(frame)
            dst_mtx = dst(frame[:, 3:6], frame[:, 3:6])
            r, c = np.where(dst_mtx == np.max(dst_mtx))
            # print(r,c)
            # print(frame[r[0]])
            # print(frame[c[0]])
            direction = frame[r[0]] - frame[c[0]]
            directions.append(direction[3:6])
            # print(directions)
        directions = np.asarray(directions)
        data_to_write = pd.DataFrame(data_to_write, dtype=float)
        data_to_write.insert(loc=6, column='6', value=temp)
        for n in range(int((num_elements_with_tile-1) / num_atom)):
            for m in range(num_atom):
                #print(n,m)
                if m != 0:
                    data_to_write = data_to_write.drop(labels=[n*num_atom+m], axis=0)
        data_to_write = data_to_write.values

        original_list = ["C"]
        symbol = [item for sublist in [[item] * len(data_to_write) for item in original_list] for item in sublist]

        print(cell_data.iloc[5+j*(len(data_to_write)*num_atom+9):6+j*(len(data_to_write)*num_atom+9), :])

        cell1 = cell_data.iloc[5+j*(len(data_to_write)*num_atom+9):6+j*(len(data_to_write)*num_atom+9), 1:2].astype(float).values - cell_data.iloc[5+j*(len(data_to_write)*num_atom+9):6+j*(len(data_to_write)*num_atom+9), 0:1].astype(float).values
        cell2 = cell_data.iloc[6+j*(len(data_to_write)*num_atom+9):7+j*(len(data_to_write)*num_atom+9), 1:2].astype(float).values - cell_data.iloc[6+j*(len(data_to_write)*num_atom+9):7+j*(len(data_to_write)*num_atom+9), 0:1].astype(float).values
        cell3 = cell_data.iloc[7+j*(len(data_to_write)*num_atom+9):8+j*(len(data_to_write)*num_atom+9), 1:2].astype(float).values - cell_data.iloc[7+j*(len(data_to_write)*num_atom+9):8+j*(len(data_to_write)*num_atom+9), 0:1].astype(float).values

        cell = [cell1.reshape(1).item(), cell2.reshape(1).item(), cell3.reshape(1).item()]
        print(cell)

        structure = Atoms(symbols=symbol, positions=data_to_write[:, 3:6], cell=cell)

        # print(structure)

        soap = SOAP(
            species=["C"],
            periodic=True,
            r_cut=50,
            n_max=6,
            l_max=6,
            sigma=0.5,
            sparse=False
        )

        feature = soap.create(structure)
        output = np.hstack((data_to_write, directions, feature))
        output = output[:500, :]

        mbtr1 = MBTR(
            species=["C"],
            geometry={"function": "distance"},
            grid={"min": 2, "max": 17, "n": 57, "sigma": 0.2},
            weighting={"function": "exp", "r_cut": 15, "threshold": 1e-3},
            periodic=True,
            normalization="l2",
        )

        mbtr2 = MBTR(
            species=["C"],
            geometry={"function": "angle"},
            grid={"min": 0, "max": 200, "n": 100, "sigma": 2.5},
            weighting={"function": "exp", "r_cut": 12, "threshold": 1e-3},
            periodic=True,
            normalization="l2",
        )

        M1 = mbtr1.create(structure)

        M2 = mbtr2.create(structure)

        M = np.hstack((M1, M2))
        out = np.vstack((output, M))

        with open(f'/gpfs/ipfs/home/jingsiliao/MPDATASET/{id}/{id}_{order}_{melt}_{temper}K_{j}.txt', 'w') as f:
            np.savetxt(f, out, fmt='%.6f',delimiter=',')


    return data_to_write, num_dataframes

os.mkdir(f'/gpfs/ipfs/home/jingsiliao/MPDATASET/{id}')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart}k/dump_{crytempstart}K.dump', n_atom, id, 'order', melt, crytempstart)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart}K_{i+201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart}K_{i+801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart}K_{i+1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+10}k/dump_{crytempstart+10}K.dump', n_atom, id, 'order', melt, crytempstart+10)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+10}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+10}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+10}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+20}k/dump_{crytempstart+20}K.dump', n_atom, id, 'order', melt, crytempstart+20)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+20}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+20}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+20}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+30}k/dump_{crytempstart+30}K.dump', n_atom, id, 'order', melt, crytempstart+30)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+30}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+30}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{crytempstart+30}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/liq{liqtempstart}k/dump_{liqtempstart}K.dump', n_atom, id, 'disorder', melt, liqtempstart)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/liq{liqtempstart+10}k/dump_{liqtempstart+10}K.dump', n_atom, id, 'disorder', melt, liqtempstart+10)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+10}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+10}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+10}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/liq{liqtempstart+20}k/dump_{liqtempstart+20}K.dump', n_atom, id, 'disorder', melt, liqtempstart+20)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+20}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+20}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+20}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/liq{liqtempstart+30}k/dump_{liqtempstart+30}K.dump', n_atom, id, 'disorder', melt, liqtempstart+30)
print(dataframes)
print(num_dataframes)

for i in range(600):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_train.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+30}K_{i + 201}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_train.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_valid.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+30}K_{i + 801}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_valid.txt', 'a') as f:
        f.write(f'{melt}\n')

for i in range(200):
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_test.txt', 'a') as f:
        f.write(f'{id}_order_{melt}_{liqtempstart+30}K_{i + 1001}\n')
    with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/tc_list_test.txt', 'a') as f:
        f.write(f'{melt}\n')

with open('/gpfs/ipfs/home/jingsiliao/MPDATASET/crystal_id_list.txt', 'a') as f:
    f.write(f'{id}\n')