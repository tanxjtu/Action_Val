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
import glob

parser = argparse.ArgumentParser(description='Action Recognition')
#   data info
parser.add_argument('--dataname', default='Charades', type=str, help='Data name [Charades] ')

parser.add_argument('--batch_size', default=45, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser.add_argument('--stride', default=60, type=int, help='sample stride for test video')

# net info

parser.add_argument('--in_fm_sz', default=224, type=int, help='frame size in to net in order to cal scale for ROI')

parser.add_argument('--fm_us', default=16, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

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


def vid_map(Video_loader, model, print_freq=10):
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
        if i % int(0.15*len(Video_loader)) == 0:
            print(i, ' / ', len(Video_loader))
        # print(i)
        # if i ==10:
        #     print(i)
        #     return out_dict
    return out_dict


def write_log(file_dir, file):

    if not os.path.exists(file_dir):
        with open(file_dir, 'w') as f:
            f.write(file)
    else:
        with open(file_dir, 'a') as f:
            f.write(file)


if __name__ == '__main__':

    # ===========some info not need to add into args =====================
    Charades_val_CSV = './data/Charades_v1_test.csv'
    data_dir = '/VIDEO_DATA/Charades_v1_rgb'  # dir store the rgb frames
    data_workers = 8
    dir_name = '../'
    # ===========some info not need to add into args =====================

    # -----------------add model list ------------
    model_lt_dir = '/home/thl/Desktop/Video_Action/sv_model/11_Charades_R3D/model'
    model_list = glob.glob(model_lt_dir+'/*')
    model_list.sort()
    model = 'R3D'
    file_dir = './out_result/' + model + '_load_onlyt)C4_lr15.txt'
    model_stride = 2

    # -----------------add model list ------------

    # ----------------best mAP---------------------
    best_mAP = 0
    best_model = model_list[0]
    # ----------------best mAP---------------------

    if model == 'I3D':
        test_transforms = transforms.Compose([videotransform.IMG_resize(256, 256),
                                              videotransform.CenterCrop(args.in_fm_sz),
                                              videotransform.ToTensor()])  # contain normalize
        val_video_data = Charades_VD(data_info_scv=Charades_val_CSV, data_root=data_dir, mode='rgb', fm_us=args.fm_us,
                                     transforms=test_transforms, stride=args.stride)
    elif model == 'R3D':
        test_transforms = transforms.Compose([videotransform.IMG_resize(256, 256),
                                              videotransform.CenterCrop_R3D(args.in_fm_sz),
                                              videotransform.Normalize_R3D([114.7748, 107.7354, 99.4750], [1, 1, 1])])
        val_video_data = Charades_VD(data_info_scv=Charades_val_CSV, data_root=data_dir, mode='rgb', fm_us=args.fm_us,
                                     transforms=test_transforms, stride=args.stride)
        Video_loader = torch.utils.data.DataLoader(val_video_data, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=data_workers, pin_memory=True)
        for i in args.__dict__.keys():
            print(i, ':\t', args.__dict__[i])

    print('Find {} train samples '.format(len(val_video_data)))

    achieved_model = []
    model_name = model_list[0]  # define by user !

    while 1:
        if model_name not in achieved_model:

            if os.path.exists(model_name):

                print('Process Model', model_name.split('/')[-1])
                sleep = 0

                if model == 'I3D':
                    I3D = InceptionI3d(400, in_channels=3, dropout_keep_prob=args.dropout, phase='eval', in_size=args.in_fm_sz,
                                       fm_use=args.fm_us)
                    I3D.replace_logits(157)
                    I3D.load_state_dict(torch.load(model_name))
                    I3D = nn.DataParallel(I3D).cuda()
                    Model = I3D
                elif model == 'R3D':
                    R3D = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32, sample_size=args.in_fm_sz,
                                            sample_duration=args.fm_us, phase='eval')
                    R3D.replace_logits(157)
                    R3D.load_state_dict(torch.load(model_name))
                    R3D = nn.DataParallel(R3D).cuda()
                    Model = R3D

                VD_pred = vid_map(Video_loader, Model, print_freq=10)
                Map = winsmooth(VD_pred)
                mAP, _, ap = map.charades_map(np.vstack(Map[0]), np.vstack(Map[1]))
                print('mAp: {}, model name: {}'.format(round(mAP, 6), model_name.split('/')[-1]))

                write_log(file_dir, str(round(mAP, 6))+'             '+ model_name.split('/')[-1].split('.')[0]+'\n')
                print('Data Saved in ', file_dir)
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_model = model_name
                print('The bast mAP: {} Bast model: {}'.format(round(best_mAP, 6), best_model.split('/')[-1]))

                achieved_model.append(model_name)
                Next_model_No = int(model_name.split('/')[-1][3:6]) + model_stride
                Next_model_name = model + "%03d_%s" % (Next_model_No, args.dataname + "_model.pth")
                Next_model_Dir = model_lt_dir + '/' + Next_model_name
                model_name = Next_model_Dir

            else:
                time.sleep(30)
                sleep = sleep + 1
                if sleep == 1:
                    print('Sleep to wait ')



