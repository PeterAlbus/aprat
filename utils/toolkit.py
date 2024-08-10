import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    # if x == 0: 
    #     return x
    # added for  drop_last=True in coda_prompt
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, base=10, increment=10, 
             cls_num_list=None, thres_m = 100, thres_f = 20):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    
    # Each accuracy
    acc_list_str = []
    for class_id in range(0, np.max(y_true)+1):
        idxes = np.where(y_true == class_id)[0]
        acc_ = np.around(
            (y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2
        )
        acc_list_str.append(str(acc_))
    
 
    # head, median, tail
    
    idx_head = cls_num_list > thres_m
    idx_tail = cls_num_list < thres_f
    idx_med = (1 - idx_head - idx_tail).astype(np.bool_)
    # print(idx_med)
    acc_list = np.array([eval(acc) for acc in acc_list_str])
    # bug---------------------------
    all_acc["h-m-f"] = (acc_list[idx_head].mean(), acc_list[idx_med].mean(), acc_list[idx_tail].mean())
    all_acc["separate"] = " ".join(acc_list_str)
    # Grouped accuracy

    start = 0
    end = base 
    print(np.max(y_true))
    while end <= np.max(y_true)+1:
        idxes = np.where(
            np.logical_and(y_true >= start, y_true < end)
        )[0]
        label = "{}-{}".format(
            str(start).rjust(2, "0"), str(end).rjust(2, "0")
        )
        start = end
        end += increment
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # for class_id in range(0, np.max(y_true), increment):
    #     idxes = np.where(
    #         np.logical_and(y_true >= class_id, y_true < class_id + increment)
    #     )[0]
    #     label = "{}-{}".format(
    #         str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
    #     )
    #     all_acc[label] = np.around(
    #         (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    #     )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = 0 if len(idxes) == 0 else np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])


    return np.array(images), np.array(labels)
