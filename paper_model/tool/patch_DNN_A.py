import pandas as pd
import numpy as np

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
    atoms = pd.read_csv('./cry500/model.data', sep=' ', names=['id', 'mol', 'type', 'q', 'x', 'y', 'z', 'u1', ], skiprows=1)
    atoms = atoms.dropna()
    atoms = atoms.drop(labels=['id', 'mol', 'type', 'q'], axis=1)
    atoms = atoms.values
    atoms = np.asarray(atoms, dtype=float)
    print(atoms.shape)

    frame_data = atoms.reshape((-1, n_atom, 3))
    print(frame_data)
    print(frame_data[0])

    with open('./cry500/cry500.pdb', 'r', encoding='gbk') as f:
        # 读取所有行
        lines = f.readlines()
        # 获取第3行到第6行的数据
        desired_lines = lines[6:9]

    data = [line.strip().split() for line in desired_lines]  # 去除每行前后的空白并分割成列表
    print(data)

    matrix = [row[1:4] for row in data]
    print(matrix)

    matrix = np.asarray(matrix, dtype=float)  # 将列表转换为numpy数组，并指定数据类型为float
    print(type(matrix))

    print(matrix)

    scale_atoms = np.dot(atoms, matrix)

    ori_directions = []
    for frame in frame_data:
        # print(frame)
        frame = np.dot(frame, matrix)
        dst_mtx = dst(frame, frame)
        r, c = np.where(dst_mtx == np.max(dst_mtx))
        # print(r,c)
        ori_direction = frame[r[0]] - frame[c[0]]
        ori_directions.append(ori_direction / np.linalg.norm(ori_direction))
        # print(directions)
    ori_directions = np.asarray(ori_directions, dtype=float)

    data_to_get = file
    data_to_write = []
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
    data = data.drop(labels=['atom', 'mass', 'q',
                             'vx', 'vy', 'vz',
                             'fx', 'fy', 'fz',
                             'pe1', 'pe2',
                             'stress11', 'stress12', 'stress13',
                             'stress14', 'stress15', 'stress16',
                             'stress21', 'stress22', 'stress23',
                             'stress24', 'stress25', 'stress26',
                             'xs', 'ys', 'zs'], axis=1)

    #data = data[['id', 'atom', 'mass', 'q', 'vx', 'vy', 'vz', 'xs', 'ys', 'zs']]
    data = data.reset_index(drop=True)
    print(data)
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

        data_to_write = data_to_write.reset_index(drop=True)  # 重编号
        data_to_write = data_to_write.values
        print(data_to_write.shape)
        ori_xyz = np.asarray(data_to_write[:, 3:6], dtype=float)
        print(type(ori_xyz))
        print(type(matrix))
        print(matrix)
        data_xyz = np.dot(ori_xyz, matrix)
        displace = data_xyz - scale_atoms

        frame_data = data_to_write.reshape((-1, num_atom, 6))
        # print(frame_data.shape)
        directions = []
        for frame in frame_data:
            print(frame)
            frame_xyz = np.asarray(frame[:, 3:6], dtype=float)
            norm_xyz = np.dot(frame_xyz, matrix)
            dst_mtx = dst(norm_xyz, norm_xyz)
            r, c = np.where(dst_mtx == np.max(dst_mtx))
            print(r,c)
            print(frame[r[0]])
            print(frame[c[0]])
            direction = norm_xyz[r[0]] - norm_xyz[c[0]]
            if direction[0] > 0.7:
                direction[0] = direction[0] - 1
            elif direction[0] < -0.7:
                direction[0] = direction[0] + 1
            elif direction[1] > 0.7:
                direction[1] = direction[1] - 1
            elif direction[1] < -0.7:
                direction[1] = direction[1] + 1
            elif direction[2] > 0.7:
                direction[2] = direction[2] - 1
            elif direction[2] < -0.7:
                direction[2] = direction[2] + 1
            directions.append(direction/np.linalg.norm(direction))
            # print(directions)
        directions = np.asarray(directions)
        directions_displace = directions - ori_directions

        directions = np.repeat(directions, n_atom, axis=0)
        directions_displace = np.repeat(directions_displace, n_atom, axis=0)

        data_to_write = np.hstack((data_to_write, displace, directions, directions_displace))
        print(data_to_write.shape)

        data_to_write = pd.DataFrame(data_to_write, dtype=float, columns=['id', 'mol', 'ke',
                                                                           'x', 'y', 'z',
                                                                          'dx', 'dy', 'dz',
                                                                          'drx', 'dry', 'drz',
                                                                          'ddrx', 'ddry', 'ddrz'])
        print(data_to_write)
        data_to_write = data_to_write.sort_values(by=['mol', 'id'])
        data_to_write = data_to_write.reset_index(drop=True)


        for n in range(int((num_elements_with_tile-1) / num_atom)):
            for m in range(num_atom):
                #print(n,m)
                if m != 0:
                    data_to_write = data_to_write.drop(labels=[n*num_atom+m], axis=0)

        data_to_write = data_to_write.values
        data_to_write = np.asarray(data_to_write, dtype=float)

        output = data_to_write[:500, :]

        for patch in output:
            if patch[6] > 0.7:
                patch[6] = patch[6] - 1
            elif patch[6] < -0.7:
                patch[6] = patch[6] + 1
            elif patch[7] > 0.7:
                patch[7] = patch[7] - 1
            elif patch[7] < -0.7:
                patch[7] = patch[7] + 1
            elif patch[8] > 0.7:
                patch[8] = patch[8] - 1
            elif patch[8] < -0.7:
                patch[8] = patch[8] + 1

        output[:, 6:9] = np.dot(output[:, 6:9], np.linalg.inv(matrix))

        output = output[:, 6:]
        D = np.zeros(9)

        output = np.vstack((output, D))

        with open(f'/gpfs/ipfs/home/jingsiliao/MPDATASET/{id}/{id}_{order}_{melt}_{temper}K_{j}.txt', 'r') as f:
            # 读取所有行
            lines = f.readlines()

        ori_data = [line.strip().split(',') for line in lines]  # 去除每行前后的空白并分割成列表
        # print(data)

        ori_data = np.asarray(ori_data, dtype=float)  # 将列表转换为numpy数组，并指定数据类型为float
        out = np.hstack((ori_data, output))

        print(out.shape)

        with open(f'/gpfs/ipfs/home/jingsiliao/MPDATASET/{id}/{id}_{order}_{melt}_{temper}K_{j}.txt', 'w') as f:
            np.savetxt(f, out, fmt='%.5f',delimiter=',')


    return data_to_write, num_dataframes

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart}k/dump_{crytempstart}K.dump', n_atom, id, 'order', melt, crytempstart)
print(dataframes)
print(num_dataframes)

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+10}k/dump_{crytempstart+10}K.dump', n_atom, id, 'order', melt, crytempstart+10)
print(dataframes)
print(num_dataframes)

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+20}k/dump_{crytempstart+20}K.dump', n_atom, id, 'order', melt, crytempstart+20)
print(dataframes)
print(num_dataframes)

dataframes, num_dataframes = read_dump(f'/gpfs/ipfs/home/jingsiliao/CRYDATA/{id}/cry{crytempstart+30}k/dump_{crytempstart+30}K.dump', n_atom, id, 'order', melt, crytempstart+30)
print(dataframes)
print(num_dataframes)