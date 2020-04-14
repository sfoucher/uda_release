from __future__ import print_function
from __future__ import division
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.utils.data as torchdata

from models.ResNet import ResNetCifar as ResNet
from dset_loaders.prepare_dataset import prepare_dataset

from utils.train import train
from utils.parse_tasks import parse_tasks
from utils.SSHead import extractor_from_layer3
from utils.plot_all_epoch_stats import plot_all_epoch_stats
from utils.misc import *



def launch_train(args):
    image_size = 32
    num_workers= 1
    if (args.source=='usps' and args.target=='mnist') or (args.source=='mnist' and args.target=='usps'):
        channels = 1
    else:
        channels = 3
    if (args.source=='cifar' and args.target=='stl') or (args.source=='stl' and args.target=='cifar'):
        classes = 9
    else:
        classes = 10

    print('==> Building model..')
    net = ResNet(args.depth, args.width, classes=classes, channels=channels).cuda()
    ext = extractor_from_layer3(net)

    print('==> Preparing datasets..')
    sc_tr_dataset, sc_te_dataset = prepare_dataset(args.source, image_size, channels, path=args.data_root)
    sc_tr_loader = torchdata.DataLoader(sc_tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers= num_workers, pin_memory=True)
    sc_te_loader = torchdata.DataLoader(sc_te_dataset, batch_size=args.batch_size, shuffle=False, num_workers= num_workers, pin_memory=True)

    tg_tr_dataset, tg_te_dataset = prepare_dataset(args.target, image_size, channels, path=args.data_root)
    tg_te_loader = torchdata.DataLoader(tg_te_dataset, batch_size=args.batch_size, shuffle=False, num_workers= num_workers, pin_memory=True)
    sstasks = parse_tasks(args, ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset)

    criterion = nn.CrossEntropyLoss().cuda()
    parameters = list(net.parameters())
    for sstask in sstasks:
        parameters += list(sstask.head.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

    all_epoch_stats = []
    print('==> Running..')
    for epoch in range(1, args.nepoch+1):
        print('Source epoch %d/%d lr=%.3f' %(epoch, args.nepoch, optimizer.param_groups[0]['lr']))
        print('Error (%)\t\tmmd\ttarget test\tsource test\tunsupervised test')


        epoch_stats = train(args, net, ext, sstasks,
            criterion, optimizer, scheduler, sc_tr_loader, sc_te_loader, tg_te_loader, epoch)
        all_epoch_stats.append(epoch_stats)
        torch.save(all_epoch_stats, args.outf + '/loss.pth')
        plot_all_epoch_stats(all_epoch_stats, args.outf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --batch_size 16 --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --milestone_1 50 --milestone_2 75 --rotation --outf D:/DEV/uda_release/output/cifar_stl_r
    parser.add_argument('--source', default= 'cifar10', required=True)
    parser.add_argument('--target', default= 'stl10', required=True)
    ################################################################
    parser.add_argument('--nepoch', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--milestone_1', default=5, type=int)
    parser.add_argument('--milestone_2', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_batches_per_test', default=1000, type=int)
    ################################################################
    parser.add_argument('--rotation', default=True, action='store_true')
    parser.add_argument('--lr_rotation', default=0.1, type=float)
    parser.add_argument('--quadrant', action='store_true')
    parser.add_argument('--lr_quadrant', default=0.1, type=float)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--lr_flip', default=0.1, type=float)
    ################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=16, type=int)
    parser.add_argument('--outf', default='C:/DEV/uda_release/output/cifar_stl_r')
    parser.add_argument('--data_root', default='C:/DEV/uda_release/data/datasets/')
    args = parser.parse_args()
    my_makedir(args.outf)
    cudnn.benchmark = False
    launch_train(args)