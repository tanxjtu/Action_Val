import os
import time
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time
import torch.utils.data
import videotransform
from Charades import Charades, Charades_VD
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from pytorch_i3d import InceptionI3d
import map

from torch.autograd import Variable

parser = argparse.ArgumentParser(description='I3D Action Recognition')
#   data info
parser.add_argument('--dataname', default='Charades', type=str, help='Data name [Charades] ')

parser.add_argument('--batch_size', default=60, type=int, help='5 for every GPU (NVIDIA 1080Ti)')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# net info
parser.add_argument('--model_pth', default='../model/I3D08_Charades_model.pth', metavar='DIR to save pth')

args = parser.parse_args()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_log(args, lr, loss, epoch, base_dir):
    # txt_log_Dir = os.path.join(args.txtlog, args.dataname, args.arch + str(epoch)+'.txt')
    txt_log_Dir = os.path.join(base_dir, args.arch + str(epoch)+'.txt')
    file = open(txt_log_Dir, mode='w')
    for i in args.__dict__.keys():
        file.write(i +'\t' + str(args.__dict__[i])+'\n')
    file.write('lr   :' + str(lr)+'\n')
    file.write('Avg_loss   :' + str(loss) + '\n')
    file.close()
    print('The train log saved to :', txt_log_Dir)


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
            out_vid.append(clp_out.max(0))
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
    data_time = AverageMeter()
    model.eval()

    end = time.time()
    out_dict = {}
    for i, (imgs, label, V_id, st_f, Scene) in enumerate(Video_loader):
        # input = imgs.float().cuda(async=True)
        input = imgs
        input_var = torch.autograd.Variable(imgs.float().cuda(async=True), volatile=True)
        # input_var = torch.autograd.Variable(input,volatile=True)
        output, out_logits = model(input_var)
        t = input.size(2)
        C_logits = F.upsample(out_logits, t, mode='linear')
        # C_logits = F.upsample(out_logits, t, mode='linear')
        # C_logits_d = C_logits.data.cpu().numpy()   # v1
        C_logits_d = output.data.cpu().numpy()   # v2
        batch_sz = imgs.shape[0]
        for j in range(batch_sz):
            if V_id[j] in out_dict.keys():
                out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :], st_f[j]))
            else:
                out_dict[V_id[j]] = {'out':[], 'label':None}
                out_dict[V_id[j]]['label'] = label[j,:,0].cpu().numpy().astype(np.uint8)
                out_dict[V_id[j]]['out'].append((C_logits_d[j, :, :], st_f[j]))
        print(i,' / ', len(Video_loader))
        if i ==20:
            return  out_dict
    return out_dict


if __name__ == '__main__':

    # ===========some info not need to add into args =====================
    Charades_val_CSV = '/home/thl/Desktop/study/I3D_Charades/data/Charades/Charades_v1_test.csv'
    data_dir = '/VIDEO_DATA/Charades_v1_rgb'                      # dir store the rgb frames
    data_workers = 30
    print_freq = 10
    epoch = int(args.model_pth[-21:-19])
    dir_name = '../'
    # ===========some info not need to add into args =====================
    test_transforms = transforms.Compose([videotransform.IMG_resize(224, 224),
                                          videotransform.ToTensor()])  # contain normalize

    val_video_data = Charades_VD(data_info_scv=Charades_val_CSV, data_root=data_dir,
                                 mode='rgb', transforms=test_transforms)

    Video_loader = torch.utils.data.DataLoader(val_video_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=data_workers, pin_memory=True)

    print('Find {} train samples '.format(len(val_video_data)))

    for i in args.__dict__.keys():
        print(i, ':\t', args.__dict__[i])

    I3D = InceptionI3d(400, in_channels=3, dropout_keep_prob=0)
    I3D.replace_logits(157)
    # I3D = nn.DataParallel(I3D).cuda()
    I3D.load_state_dict(torch.load('../model/rgb_charades.pt'))
    I3D = nn.DataParallel(I3D).cuda()
    # I3D.load_state_dict(torch.load(args.model_pth))

    VD_pred = vid_map(Video_loader, I3D, epoch, print_freq=10)

    output = []
    gt = []

    Map = winsmooth(VD_pred)
    mAP, _, ap = map.charades_map(np.vstack(Map[0]), np.vstack(Map[1]))
    print ( 'The final mAp is:', mAP)
    submission_file(
        Map[2], Map[0], '{}/epoch_{:03d}.txt'.format(dir_name + 'vallog', epoch + 1))
