import numpy as np
import os
import matplotlib.pyplot as plt


VALID_CLASSES = ['Car', 'Van', 'Pedestrian', 'Person_sitting', 'Cyclist', 'DontCare']
CLS_DICT = {'Car':0, 'Pedestrian':1, 'Cyclist':2}

MIN_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
MIN_OVERLAP = {'Car':0.7,'Pedestrian':0.5,'Cyclist': 0.5}
N_SAMPLE_PTS = 41


def load_gt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    record_list = []

    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0:
            continue
        if line[0] not in VALID_CLASSES:
            continue

        record = {}
        record['class'] = line[0]
        record['trunc'] = float(line[1])
        record['occ'] = float(line[2])
        record['box'] = [float(c) for c in line[4:8]]
        record_list.append(record)
    return record_list



def load_pred(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    record_list = []

    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 0:
            continue
        if line[0] not in VALID_CLASSES:
            continue

        record = {}
        record['class'] = line[0]
        record['box'] = [float(c) for c in line[4:8]]
        record['score'] = float(line[-1])
        record_list.append(record)
    return record_list


def get_thresholds(v, n_groundTruth):
    v = np.array(v)
    sort_ind_desc = np.argsort(v * -1)
    vs = v[sort_ind_desc]

    t = []
    current_recall = 0

    for i in range(vs.shape[0]):
        l_recall = (i+1)/n_groundTruth

        if i < vs.shape[0] - 1:
            r_recall = (i+2)/n_groundTruth
        else:
            r_recall = l_recall

        if (r_recall - current_recall) < (current_recall - l_recall) and i < (vs.shape[0] - 1):
            continue
        t.append(vs[i])
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)
    return t


def get_iou(gt, pred, union=True):
    gxmin, gymin, gxmax, gymax = gt['box']
    pxmin, pymin, pxmax, pymax = pred['box']

    ixmin = np.maximum(gxmin, pxmin)
    iymin = np.maximum(gymin, pymin)
    ixmax = np.minimum(gxmax, pxmax)
    iymax = np.minimum(gymax, pymax)

    ih = np.maximum(0., iymax - iymin)
    iw = np.maximum(0., ixmax - ixmin)

    gvol = (gxmax - gxmin) * (gymax - gymin)
    pvol = (pxmax - pxmin) * (pymax - pymin)
    ivol = iw * ih

    if union:
        iou = ivol / (gvol + pvol - ivol)
    else:
        iou = ivol / pvol
    return iou


def clean_data(gts, preds, cls, diff):
    ignore_gt = []
    ignore_pred = []
    dontcare = []

    n_gt = 0

    #clean ground truth
    for gt in gts:
        #set ignore
        if cls == gt['class']:
            valid_class = 1
        else:
            if gt['class'] == 'Van' and cls == 'Car':
                valid_class = 0
            elif gt['class'] == 'Person_sitting' and cls == 'Pedestrian':
                valid_class = 0
            else:
                valid_class = -1

        height = gt['box'][3] - gt['box'][1]

        if gt['occ'] > MAX_OCCLUSION[diff] or gt['trunc'] > MAX_TRUNCATION[diff] or height < MIN_HEIGHT[diff]:
            ignore = True
        else:
            ignore = False

        if valid_class == 1 and not ignore:
            n_gt += 1
            ignore_gt.append(0)
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignore_gt.append(1)
        else:
            ignore_gt.append(-1)

        #set Don't care
        if gt['class'] == 'DontCare':
            dontcare.append(True)
        else:
            dontcare.append(False)

    #clean predictions
    for pred in preds:
        if pred['class'] == cls:
            valid_class = 1
        else:
            valid_class = 0
        height = pred['box'][3] - pred['box'][1]

        if height < MIN_HEIGHT[diff]:
            ignore_pred.append(1)
        elif valid_class == 1:
            ignore_pred.append(0)
        else:
            ignore_pred.append(-1)

    return ignore_gt, dontcare, ignore_pred, n_gt



def compute_statistics(gts, preds, dontcare, ignore_gt, ignore_pred, compute_fp, threshold, cls, diff):
    n_gt = len(gts)
    n_pred = len(preds)

    assigned_detection = [False for _ in range(n_pred)]
    TP, FP, FN = 0, 0, 0
    vs = []

    ignore_threshold = []
    if compute_fp:
        for pred in preds:
            if pred['score'] < threshold:
                ignore_threshold.append(True)
            else:
                ignore_threshold.append(False)
    else:
        for pred in preds:
            ignore_threshold.append(False)

    for i in range(n_gt):
        if ignore_gt[i] == -1:
            continue

        det_idx = -1
        valid_detection = -1
        max_iou = 0.
        assigned_ignored_det = False

        for j in range(n_pred):
            if ignore_pred[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignore_threshold[j]:
                continue

            iou = get_iou(gts[i], preds[j])

            if not compute_fp and iou > MIN_OVERLAP[cls] and preds[j]['score'] > threshold:
                det_idx = j
                valid_detection = preds[j]['score']
            elif compute_fp and iou > MIN_OVERLAP[cls] and (iou > max_iou or assigned_ignored_det) and ignore_pred[j] == 0:
                max_iou = iou
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif compute_fp and iou > MIN_OVERLAP[cls] and valid_detection == -1. and ignore_pred[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if valid_detection == -1 and ignore_gt[i] == 0:
            FN += 1
        elif valid_detection != -1 and (ignore_gt[i] == 1 or ignore_pred[det_idx]==1):
            assigned_detection[det_idx] = True
        elif valid_detection != -1:
            TP += 1
            vs.append(preds[det_idx]['score'])
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(n_pred):
            if not (assigned_detection[i] or ignore_pred[i]==-1 or ignore_pred[i]==1 or ignore_threshold[i]):
                FP += 1

        n_stuff = 0
        for i in range(n_gt):
            if not dontcare[i]:
                continue
            for j in range(n_pred):
                if assigned_detection[j]:
                    continue
                if ignore_pred[j] == -1 or ignore_pred[j] == 1:
                    continue
                if ignore_threshold[j]:
                    continue
                iou = get_iou(preds[j], gts[i], union=False)
                if iou > MIN_OVERLAP[cls]:
                    assigned_detection[j] = True
                    n_stuff += 1

        FP -= n_stuff

    return TP, FP, FN, vs


def eval_class(gt_list, pred_list, cls, diff):
    ignore_gt_list = []
    ignore_pred_list = []
    dontcare_list = []
    total_gt_num = 0

    #clean data
    vs = []
    for i in range(len(gt_list)):
        ignore_gt, dontcare, ignore_pred, n_gt_ = clean_data(gt_list[i], pred_list[i], cls, diff)
        ignore_gt_list.append(ignore_gt)
        ignore_pred_list.append(ignore_pred)
        dontcare_list.append(dontcare)
        total_gt_num += n_gt_

        _, _, _, vs_ = compute_statistics(gt_list[i], pred_list[i], dontcare, ignore_gt, ignore_pred, False, 0, cls, diff)
        vs = vs + vs_
    thresholds = get_thresholds(vs, total_gt_num)
    len_th = len(thresholds)
    TPs = [0.] * len_th
    FPs = [0.] * len_th
    FNs = [0.] * len_th

    for i in range(len(gt_list)):
        for t, th in enumerate(thresholds):
            TP, FP, FN, _, = compute_statistics(gt_list[i], pred_list[i], dontcare_list[i], ignore_gt_list[i], ignore_pred_list[i], True, th, cls, diff)
            TPs[t] += TP
            FPs[t] += FP
            FNs[t] += FN

    precisions = [0.] * N_SAMPLE_PTS
    recalls = []

    for t, th in enumerate(thresholds):
        r = TPs[t] / (TPs[t] + FNs[t])
        recalls.append(r)
        precisions[t] = TPs[t] / (TPs[t] + FPs[t])

    for t, th in enumerate(thresholds):
        precisions[t] = np.max(precisions[t:])

    return  precisions, recalls


def plot_and_compute(precisions,cls, plot):
    if plot:
        Xs = np.arange(0., 1., 1./len(precisions[0]))

        l_easy = plt.plot(Xs, precisions[0], c='green')[0]
        l_moderate = plt.plot(Xs, precisions[1], c='blue')[0]
        l_hard = plt.plot(Xs, precisions[2], c='red')[0]

        labels = ['Easy','Moderate','Hard']
        plt.legend(handles=[l_easy,l_moderate,l_hard],labels=labels,loc='best')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(cls)
        plt.ylim((0,1.0))
        plt.grid()
        plt.savefig('2d_result.png')
        plt.show()
        plt.close()

    val_easy, val_moderate, val_hard = 0., 0., 0.
    for i in range(0, N_SAMPLE_PTS,4):
        val_easy += precisions[0][i]
        val_moderate += precisions[1][i]
        val_hard += precisions[2][i]

    ap_easy = 100. * val_easy / 11.
    ap_moderate = 100. * val_moderate / 11.
    ap_hard = 100. * val_hard / 11.

    print('2D Detection AP for %s\n'%cls)
    print('Easy: %f'%ap_easy)
    print('Moderate: %f'%ap_moderate)
    print('Hard: %f'%ap_hard)




def eval(gt_dir, pred_dir, cls):
    gt_list = []
    pred_list = []

    for f in os.listdir(pred_dir):
        record_pred = load_pred(os.path.join(pred_dir, f))
        record_gt = load_gt(os.path.join(gt_dir, f))
        pred_list.append(record_pred)
        gt_list.append(record_gt)

    recall_all_diff = []
    precision_all_diff = []
    for diff in range(3):
        precisions, recalls = eval_class(gt_list, pred_list, cls, diff)
        precision_all_diff.append(precisions)
        recall_all_diff.append(recalls)

    plot_and_compute(precision_all_diff, cls)


if __name__ == '__main__':
    gt_dir = '/Users/yangchenhongyi/Downloads/kitti_eval/eval_label/'
    pred_dir = '/Users/yangchenhongyi/Documents/TEMP/result'
    #Car,  Pedestrian, Cyclist
    cls = 'Car'
    eval(gt_dir, pred_dir, cls)























