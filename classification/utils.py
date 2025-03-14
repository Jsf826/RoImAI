import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import networkx as nx


trees_SSMG = [
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 0],
    [4, 0],
    [5, 1],
    [6, 1],
    [7, 2],
    [8, 2],
    [9, 2],
    [10, 2],
    [11, 2],
]
trees_species_to_family_SSMG = [
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 0],
    [4, 0],
    [5, 1],
    [6, 1],
    [7, 2],
    [8, 2],
    [9, 2],
    [10, 2],
    [11, 2],
]


def get_order_family_target(targets, device, dataset):
    order_target_list = []
    family_target_list = []
    target_list_sig = []

    for i in range(targets.size(0)):
        family_target_list.append(int(trees_species_to_family_SSMG[targets[i]][1]))
            # elif targets[i] > 99:
            #     family_target_list.append(trees_SSMG[targets[i] - 100][1] - 1)

        target_list_sig.append(int(targets[i]))

    order_target_list = torch.from_numpy(np.array(order_target_list)).to(device)
    family_target_list = torch.from_numpy(np.array(family_target_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    return order_target_list, family_target_list, target_list_sig



def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                   y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws
