import os
import time
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import time
import torch.utils.data
from lib import videotransform
from lib.Charades import Charades, Charades_VD
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from lib.pytorch_i3d import InceptionI3d
from lib import resnext
from lib import map
import pickle
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Action Recognition')
#   data info
parser.add_argument('--dataname', default='Charades', type=str, help='Data name [Charades] ')

parser.add_argument('--batch_size', default=80, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

parser.add_argument('--stride', default=60, type=int, help='sample stride for test video')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# net info
parser.add_argument('--model_pth', default='../model/R3D056_Charades_model.pth', metavar='DIR to save pth')

parser.add_argument('--dropout', default=0.50, type=float, help='dropout Prob (default: 0.3)')

parser.add_argument('--in_fm_sz', default=224, type=int, help='frame size in to net in order to cal scale for ROI')

parser.add_argument('--fm_use', default=16, type=int, help='64 frames for every GPU (NVIDIA 1080Ti) 224x224')

args = parser.parse_args()


def winsmooth(mat, ker_nl_clip=1, ker_clips=1):  # only sommth every 64 and finally to smooth 20th finaly

    # print('applying smoothing with kernelsize {}'.format(kernelsize))
    out_vids = []
    gts = []
    name_vds = []
    mat.keys()
    for No, V_id in enumerate(mat.keys()):
        # process each video

        # add this vd label and name
        label = mat[V_id]['label']
        gts.append(label)
        name_vds.append(V_id)
        # add this vd label and name

        # loop to process each video logist
        out_vid = []
        outs_vd = mat[V_id]['out']
        time_order = [st for _, st in outs_vd]
        time_sort = np.argsort(np.array(time_order))
        # process each clip
        for i in time_sort:
            clp_out = outs_vd[i][0].T    # every clip
            temp = outs_vd[i][0].T.copy()
            n = clp_out.shape[0]
            for frm in range(n):
                st = max(0, frm-ker_nl_clip)
                ed = min(n-1, frm + ker_nl_clip)
                clp_out[frm, :] = temp[st:ed+1,:].mean(0)
            # out_vid.append(clp_out.max(0))  # original
            out_vid.append(clp_out.max(0))  # V1
        out_vids.append(np.stack(out_vid).max(0))
    return out_vids, gts, name_vds


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


def vid_map(Video_loader, model, epoch, print_freq=10):
    # data_time = AverageMeter()
    model.eval()

    out_dict = {}
    for i, (imgs, label, V_id, st_f, VD_info) in enumerate(Video_loader):
        # input = imgs.float().cuda(async=True)
        input = imgs
        input_var = torch.autograd.Variable(imgs.float().cuda(async=True), volatile=True)
        VD_info = torch.autograd.Variable(VD_info.cuda(async=True), volatile=True)
        # input_var = torch.autograd.Variable(input,volatile=True)
        output, out_logits = model(input_var, VD_info)
        t = input.size(2)
        # F.upsample(out_logits.unsqueeze(2), t, mode='bilinear')
        C_logits = F.upsample(out_logits, t, mode='linear')  #V1
        # C_logits = F.upsample(out_logits, t, mode='linear')
        C_logits_d = C_logits.data.cpu().numpy()   # v1
        # C_logits_d = output.data.cpu().numpy()   # v2
        # C_logits_d = C_logits.data.cpu().numpy()
        # C_logits_d = out_logits.data.cpu().numpy()
        batch_sz = imgs.shape[0]
        for j in range(batch_sz):
            if V_id[j] in out_dict.keys():
                out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :], st_f[j]))
            else:
                out_dict[V_id[j]] = {'out':[], 'label':None}
                out_dict[V_id[j]]['label'] = label[j,:,0].cpu().numpy().astype(np.uint8)
                out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :], st_f[j]))
        if i % 30 == 0:
            print(i, ' / ', len(Video_loader))
        # if i ==10:
        #    return out_dict
    return out_dict


if __name__ == '__main__':

    # ===========some info not need to add into args =====================
    Charades_val_CSV = './data/Charades_v1_test.csv'
    data_dir = '/VIDEO_DATA/Charades_v1_rgb'                      # dir store the rgb frames
    model_name = args.model_pth
    data_workers = 16
    print_freq = 10
    epoch = 4
    dir_name = '../'
    fm_us = args.fm_use
    # ===========some info not need to add into args =====================
    model = 'R3D'
    if model == 'I3D':
        test_transforms = transforms.Compose([videotransform.IMG_resize(256, 256),
                                              videotransform.CenterCrop(args.in_fm_sz),
                                              videotransform.ToTensor()])  # contain normalize
        val_video_data = Charades_VD(data_info_scv=Charades_val_CSV, data_root=data_dir, mode='rgb', fm_us=fm_us,
                                     transforms=test_transforms, stride=args.stride)
        I3D = InceptionI3d(400, in_channels=3, dropout_keep_prob=args.dropout, phase='eval', in_size=args.in_fm_sz,
                           fm_use=args.fm_use)
        I3D.replace_logits(157)
        I3D.load_state_dict(torch.load(model_name))
        I3D = nn.DataParallel(I3D).cuda()
        Model = I3D
    elif model == 'R3D':
        test_transforms = transforms.Compose([videotransform.IMG_resize(256, 256),
                                              videotransform.CenterCrop_R3D(args.in_fm_sz),
                                               videotransform.Normalize_R3D([114.7748, 107.7354, 99.4750], [1, 1, 1])])
        val_video_data = Charades_VD(data_info_scv=Charades_val_CSV, data_root=data_dir, mode='rgb', fm_us=fm_us,
                                     transforms=test_transforms, stride=args.stride)
        R3D = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32, sample_size=args.in_fm_sz,
                                sample_duration=args.fm_use,  phase='eval', Dense_out=False)
        # R3D = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32, sample_size=args.in_fm_sz,
        #                         sample_duration=args.fm_use, Dense_out=True)
        R3D.replace_logits(157)
        R3D.load_state_dict(torch.load(model_name))
        R3D = nn.DataParallel(R3D).cuda()
        Model = R3D
    Video_loader = torch.utils.data.DataLoader(val_video_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=data_workers, pin_memory=True)
    print('Find {} train samples '.format(len(val_video_data)))
    for i in args.__dict__.keys():
        print(i, ':\t', args.__dict__[i])

    VD_pred = vid_map(Video_loader, Model, epoch, print_freq=10)

    out_dict = './out_result/' + args.model_pth.split('/')[-1].split('.')[0] + '.pkl'
    with open(out_dict, 'wb') as file:
        pickle.dump(VD_pred, file)

    Map = winsmooth(VD_pred)
    mAP, _, ap = map.charades_map(np.vstack(Map[0]), np.vstack(Map[1]))
    print('The final mAp is:', mAP)
    submission_file(
        Map[2], Map[0], '{}/{}V2.txt'.format(dir_name + 'vallog', args.model_pth.split('/')[-1].split('.')[0]))

