import os
import cv2
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.sampler import *
from model import shake_drop_net
from cosine_optim import cosine_annealing_scheduler


def metric(logit, truth, is_average=True):

    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        if is_average:
            # top-3 accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct/len(truth)

            top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
            precision = correct[0]/1 + correct[1]/2 + correct[2]/3
            return precision, top

        else:
            return correct


def drawing_to_image(drawing, H, W):
    point = []
    time = []
    #     coordinates = []

    for t, (x, y) in enumerate(drawing):
        point.append(np.array((x, y), np.float32).T)
        time.append(np.full(len(x), t))

    point = np.concatenate(point).astype(np.float32)
    time = np.concatenate(time).astype(np.int32)

    image = np.full((H, W, 3), 0, np.uint8)
    x_max = point[:, 0].max()
    x_min = point[:, 0].min()
    y_max = point[:, 1].max()
    y_min = point[:, 1].min()
    w = x_max - x_min
    h = y_max - y_min
    # print(w,h)

    s = max(w, h)
    norm_point = (point - [x_min, y_min]) / s
    norm_point = (norm_point - [w / s * 0.5, h / s * 0.5]) * max(W, H) * 0.85
    norm_point = np.floor(norm_point + [W / 2, H / 2]).astype(np.int32)

    T = time.max() + 1
    for t in range(T):
        p = norm_point[time == t]
        x, y = p.T
        image[y, x] = 255
        N = len(p)
        for i in range(N - 1):
            x0, y0 = p[i]
            x1, y1 = p[i + 1]
            cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)

    #     return np.transpose(image, (2,0,1))
    return image


def null_augment(drawing, label, index):
    #     cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 128, 128)
    return image, label


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


def null_collate(batch):
    batch_size = len(batch)
#     cache = []
    input = []
    truth = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
#         cache.append(batch[b][2])x

    input = np.array(input).transpose(0, 3, 1, 2)
    input = torch.from_numpy(input).float()

    if truth[0] is not None:
        truth = np.array(truth)
        truth = torch.from_numpy(truth).long()

    return input, truth


def read_class():
    CLASS_NAME = []
    with open('./classes.txt', 'r') as class_file:
        for i, line in enumerate(class_file):
            line = line.rstrip('\n')
            CLASS_NAME.append(line)
    return CLASS_NAME


def read_full_df(base_path, class_array, train_split_folder='train_set', valid_split_folder='valid_set'):
    full_df = []
    train_id, valid_id = [], []
    num_class = len(class_array)
    start = timer()

    # read full df .csv files
    for i, name in enumerate(class_array):
        print('\r\t load df   :  %3d/%3d %24s  %s' % (i, num_class, name, time_to_str((timer() - start), 'sec')),
              end='', flush=True)
        name = name.replace('_', ' ')

        df = pd.read_csv(base_path + 'train_%s/%s.csv' % ('simplified', name))
        full_df.append(df)
    print('\n')

    # acquire train split set's id
    for i, name in enumerate(class_array):
        print('\r\t load train_set split:  %3d/%3d %24s  %s' % (
        i, num_class, name, time_to_str((timer() - start), 'sec')),
              end='', flush=True)
        name = name.replace('_', ' ')
        df = full_df[i]
        key_id = np.load(base_path + 'data_split/%s/%s.npy' % (train_split_folder, name))
        label = np.full(len(key_id), i, dtype=np.int64)
        drawing_id = df.loc[df['key_id'].isin(key_id)].index.values
        train_id.append(np.vstack([label, drawing_id, key_id]).T)
    train_id = np.concatenate(train_id)

    # acquire valid split set's id
    for i, name in enumerate(class_array):
        print('\r\t load valid_set split:  %3d/%3d %24s  %s' % (
        i, num_class, name, time_to_str((timer() - start), 'sec')),
              end='', flush=True)
        name = name.replace('_', ' ')
        df = full_df[i]
        key_id = np.load(base_path + 'data_split/%s/%s.npy' % (valid_split_folder, name))
        label = np.full(len(key_id), i, dtype=np.int64)
        drawing_id = df.loc[df['key_id'].isin(key_id)].index.values
        valid_id.append(np.vstack([label, drawing_id, key_id]).T)
    valid_id = np.concatenate(valid_id)

    return full_df, train_id, valid_id


class DoodleDataset(Dataset):

    def __init__(self,
                 full_df,
                 datasplit_id,
                 shuffle=False,
                 augment=null_augment,
                 mode='simplified'):
        super(DoodleDataset, self).__init__()
        assert mode in ['simplified', 'raw']

        self.datasplit_id = datasplit_id
        self.augment = augment
        self.mode = mode

        self.df = full_df
        self.id = datasplit_id

        ### shuffle
        if shuffle:
            np.random.shuffle(self.datasplit_id)
        print('\n')

    def __str__(self):
        N = len(self.id)
        string = '' \
                 + '\tmode         = %s\n' % self.mode \
                 + '\tlen(self.id) = %d\n' % N \
                 + '\n'
        return string

    def __getitem__(self, index):
        #         if self.mode == 'train':
        label, drawing_id, key_id = self.id[index]
        drawing = self.df[label]['drawing'][drawing_id]
        drawing = eval(drawing)

        #         if self.mode == 'test':
        #             label = None
        #             drawing = self.df['drawing'][index]
        #             drawing = eval(drawing)

        return self.augment(drawing, label, index)

    def __len__(self):
        return len(self.id)


if __name__ == '__main__':
    base_path = '/Volumes/JS/QuickDraw/'
    class_array = read_class()
    full_df, train_id, valid_id = read_full_df(base_path, class_array)

    train_dataset = DoodleDataset(full_df, train_id, augment=null_augment)
    train_loader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=2048,
                              drop_last=True,
                              # num_workers=2,
                              collate_fn=null_collate)

    valid_dataset = DoodleDataset(full_df, valid_id, augment=null_augment)

    valid_loader = DataLoader(valid_dataset,
                              #                           sampler=RandomSampler(valid_dataset),
                              batch_size=2048,
                              drop_last=False,
                              num_workers=2,
                              collate_fn=null_collate)

    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(train_loader))

    # log.write('** net setting **\n')
    net = shake_drop_net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    # cosine_lr_scheduler = cosine_annealing_scheduler(optimizer, 20, args.lr)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch = 10
    batch_loss = np.zeros(6, np.float32)

    # cosine_lr_scheduler.step()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        print(np.array(inputs).shape)

        outputs = net(inputs)
        print(outputs)

        loss = criterion(outputs, targets)
        precision, top = metric(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # if batch_idx % 10 == 0:
        # print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx,
        #                                                                   len(train_loader),
        #                                                                   train_loss / (batch_idx + 1),
        #                                                                   100. * correct / total))

        batch_loss[:4] = np.array((loss.item(), top[0].item(), top[2].item(), precision.item(),))
        print('%0.3f  %0.3f  %0.3f  (%0.3f) ' % (batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3]))

        if batch_idx % 10 == 0:
            torch.save(net.state_dict(),  '.results/checkpoint/%d_model.pth' % batch_idx)
            # torch.save({
            #     'optimizer': optimizer.state_dict(),
            #     'epoch': epoch,
            # },  '/checkpoint/%d_optimizer.pth' % (i))