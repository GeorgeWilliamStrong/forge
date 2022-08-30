import torch
import numpy as np


def create_s_pos(new_s_pos, bp):
    s_pos = np.zeros((new_s_pos.shape[0], 3), dtype=int)
    for i in range(new_s_pos.shape[0]):
        s_pos[i, 0] = i
        s_pos[i, 1] = new_s_pos[i, 0]+bp
        s_pos[i, 2] = new_s_pos[i, 1]+bp
    return torch.from_numpy(s_pos)


def create_hicks_s_pos(new_s_pos, m, bp, b=6.31, r=8):
    pos = new_s_pos+bp
    y = np.arange(0, m.size(0), 1)
    x = np.arange(0, m.size(1), 1)
    z = np.arange(0, r+1, 1)-r/2
    kaiser = np.kaiser(r+1, b)
    new_s_pos = []
    new_s_pos_values = []
    for i in range(pos.shape[0]):
        tempy = y-pos[i, 0]
        tempx = x-pos[i, 1]
        mesh = np.meshgrid(tempx, tempy)
        t = np.sqrt(mesh[0]**2+mesh[1]**2)
        hicks = np.interp(t, z, kaiser, left=0, right=0)*np.sinc(t)
        hicks_pos = np.transpose(np.nonzero(hicks))
        for j in range(len(hicks_pos)):
            new_s_pos.append(np.insert(hicks_pos[j],0,i))
            new_s_pos_values.append(hicks[hicks_pos[j,0], hicks_pos[j,1]])
    return torch.tensor(np.array(new_s_pos), dtype=torch.long), torch.tensor(np.array(new_s_pos_values), dtype=torch.float32)


def create_hicks_r_pos(r_pos, m, b=6.31, r=8):
    pos = r_pos.numpy()
    y = np.arange(0, m.size(0), 1)
    x = np.arange(0, m.size(1), 1)
    z = np.arange(0, r+1, 1)-r/2
    kaiser = np.kaiser(r+1, b)
    new_r_pos = []
    new_r_pos_values = []
    new_r_pos_sizes = []
    for i in range(pos.shape[0]):
        tempy = y-pos[i, 0]
        tempx = x-pos[i, 1]
        mesh = np.meshgrid(tempx, tempy)
        t = np.sqrt(mesh[0]**2+mesh[1]**2)
        hicks = np.interp(t, z, kaiser, left=0, right=0)*np.sinc(t)
        hicks_pos = np.transpose(np.nonzero(hicks))
        new_r_pos_sizes.append(len(hicks_pos))
        for j in range(len(hicks_pos)):
            new_r_pos.append(hicks_pos[j])
            new_r_pos_values.append(hicks[hicks_pos[j,0], hicks_pos[j,1]])
    return torch.tensor(np.array(new_r_pos)), torch.tensor(np.array(new_r_pos_values), dtype=torch.float32), torch.tensor(np.array(new_r_pos_sizes))


def d_hicks_to_d(d, r_pos_sizes, num_rec, num_srcs, s):
    new_d = torch.zeros(num_srcs, num_rec, len(s)).float()
    count = 0
    for j in range(len(r_pos_sizes)):
        new_d[:, j, :] = d[:, count:count+r_pos_sizes[j], :].sum(dim=1)
        count += r_pos_sizes[j]
    return new_d


def resid_hicks(resid, num_srcs, num_rec, r_pos, r_pos_sizes, r_hicks):
    if r_hicks == True:
        new_resid = torch.zeros(num_srcs, r_pos.shape[0], resid.shape[2])
        count = 0
        for i in range(num_rec):
            new_resid[:,count:count+r_pos_sizes[i]] = resid[:,i].expand(r_pos_sizes[i], num_srcs, resid.shape[2]).transpose(0,1)
            count += r_pos_sizes[i]
        return new_resid
    else:
        return resid