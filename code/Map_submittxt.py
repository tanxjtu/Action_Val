import os
import pickle
import map
import numpy as np

out_data_name = './out_result/134_228_(256_224_177).pkl'
dir_name = '../'
with open(out_data_name, 'rb') as f:
    print("Loading Pre_mAP result '%s'" % out_data_name)
    VD_pred = pickle.load(f)


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


def winsmoothV1(mat, ker_nl_clip=1, ker_clips=1):  # only sommth every 64 and finally to smooth 20th finaly

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
            clp_out = outs_vd[i][0].T  # every clip
            temp = outs_vd[i][0].T.copy()
            n = clp_out.shape[0]
            for frm in range(n):
                st = max(0, frm - ker_nl_clip)
                ed = min(n - 1, frm + ker_nl_clip)
                clp_out[frm, :] = temp[st:ed + 1, :].mean(0)
            temp_out = np.sort(clp_out, 0)

            # out_vid.append(temp_out[:-5,:].max(0))  # V1
            out_vid.append(temp_out[-5, :])
        out_vids.append(np.stack(out_vid).max(0))
    return out_vids, gts, name_vds


def winsmoothV2(mat, ker_nl_clip=1, ker_clips=1):  # only sommth every 64 and finally to smooth 20th finaly

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
            clp_out = outs_vd[i][0].T  # every clip
            temp = outs_vd[i][0].T.copy()
            n = clp_out.shape[0]
            for frm in range(n):
                st = max(0, frm - ker_nl_clip)
                ed = min(n - 1, frm + ker_nl_clip)
                clp_out[frm, :] = temp[st:ed + 1, :].mean(0)
            temp_out = np.sort(clp_out, 0)

            # out_vid.append(temp_out[:-5,:].max(0))  # V1
            out_vid.append(temp_out[-7:-3, :].mean(0))
        out_vids.append(np.stack(out_vid).max(0))
        # out_vids.append(np.sort(np.stack(out_vid), 0)[-2, :])
    return out_vids, gts, name_vds


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


Map = winsmoothV2(VD_pred)
mAP, _, ap = map.charades_map(np.vstack(Map[0]), np.vstack(Map[1]))
print( 'The final mAp is:', mAP)
submission_file(
    Map[2], Map[0], '{}/{}V2.txt'.format(dir_name + 'vallog', out_data_name.split('/')[-1].split('.')[0]))
# Map[2], Map[0], '{}/{}epoch_{:03d}.txt'.format(dir_name + 'vallog', model_name[-8:-3], epoch + 1))
