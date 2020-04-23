import random
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
# from core.utils.sampler import ClassBalancedBatchSampler

from datasets.fss import FSS



def get_dataset(name):
    """get_loader

    :param name:
    """
    return {
        "fss": FSS,
    }[name]


def get_data_path(name, config_file="config.json"):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]["data_path"]


# def split_dataset(label_path, train_size=0.8, random_state=1):

#     labels_df = pd.read_csv(label_path)
#     images = labels_df['image'].values
#     labels = labels_df['level'].values
#     num_classes = len(set(labels))

#     images_list = []
#     labels_list = []
#     for label in range(num_classes):  # num_classes
#         indexs = np.where(labels == label)
#         images_list.append(images[indexs])
#         labels_list.append(labels[indexs])

#     train_images, train_labels = [], []
#     valid_images, valid_labels = [], []
#     for label in range(num_classes):
#         X_train, X_test, y_train, y_test = train_test_split(
#             images_list[label], labels_list[label],
#             train_size=train_size,
#             random_state=random_state,
#         )
#         train_images.append(X_train)
#         train_labels.append(y_train)
#         valid_images.append(X_test)
#         valid_labels.append(y_test)

#     train_images = np.concatenate(train_images, 0)
#     train_labels = np.concatenate(train_labels, 0)
#     valid_images = np.concatenate(valid_images, 0)
#     valid_labels = np.concatenate(valid_labels, 0)

#     return [train_images, train_labels, valid_images, valid_labels]


def split_dataset_by_csv(train_label_path_list, valid_label_path, x_key='image'):
    """

    :param train_label_path:
    :param valid_label_path:
    :return:
    """
    for i, train_label_path in enumerate(train_label_path_list):
        if i == 0:
            train_labels_df = pd.read_csv(train_label_path)
        else:
            train_labels_df = train_labels_df.append(pd.read_csv(train_label_path))

    valid_labels_df = pd.read_csv(valid_label_path)

    train_class = train_labels_df[x_key].values
    # train_labels = train_labels_df[y_key].values
    valid_class = valid_labels_df[x_key].values
    # valid_labels = valid_labels_df[y_key].values

    return [train_class, valid_class]


# def make_weights_for_balanced_classes(labels, num_classes):
#     # count_dict = dict(Counter(labels))
#     total_counts, class_per_counts = get_sample_counts(list(range(num_classes)), labels, num_classes)
#     print(class_per_counts)
#     count_list = []
#     for k in sorted(class_per_counts.keys()):
#         count_list.append(class_per_counts[k])

#     weight_per_class = [0.] * num_classes

#     for i in range(num_classes):
#         weight_per_class[i] = total_counts / float(count_list[i])

#     weight = [0] * len(labels)
#     for idx, val in enumerate(labels):
#         if val <= 1:  # TODO: remove : DR 2 class 한정
#             val = 0
#         else:
#             val = 1
#         weight[idx] = weight_per_class[val]

#     return weight


def setup_dataloader(dataset_cls,
                     train_label_path_list, valid_label_path,
                     x_key=None,
                     batch_size=16,
                     num_workers=16,
                     train_aug=None,
                     valid_aug=None,
                     make_balance=None,
                     **kwargs):
    """

    :param dataset_cls:
    :param dataset:
    :param image_root:
    :param num_classes:
    :param batch_size:
    :param num_workers:
    :param train_aug:
    :param valid_aug:
    :return:
    """

    for i, train_label_path in enumerate(train_label_path_list):
        if i == 0:
            train_labels_df = pd.read_csv(train_label_path)
        else:
            train_labels_df = train_labels_df.append(pd.read_csv(train_label_path))

    valid_labels_df = pd.read_csv(valid_label_path)
    train_class = train_labels_df[x_key].values
    valid_class = valid_labels_df[x_key].values
    

    train_data = dataset_cls(train_class, augmentation=train_aug, **kwargs)
    valid_data = dataset_cls(valid_class, augmentation=valid_aug, **kwargs)

    # if make_balance:
    #     """
    #     weights = make_weights_for_balanced_classes(train_labels, num_classes)
    #     weights = torch.DoubleTensor(weights)
    #     print('len(weights):', len(weights))
    #     total_counts, class_per_counts = get_sample_counts(list(range(num_classes)), train_labels, num_classes)
    #     num_samples = int(class_per_counts[1] * 2)
    #     print('num_samples: ', num_samples)
    #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=False)
        
    #     """
    #     num_classes = kwargs.get('num_classes', 2)
    #     sampler = ClassBalancedBatchSampler(train_labels, int(batch_size/num_classes))

    #     train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=num_workers)
    #     valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader


# def get_sample_counts(class_names, labels, num_classes):
#     """

#     :param class_names:
#     :param labels:
#     :param num_classes:
#     :return: num_labels, class_pos_counts
#         27539, {'No_DR': 20240.0, 'Mild': 1935.0, 'Moderate': 4140.0, 'Severe': 686.0, 'Proliferative_DR': 538.0}
#     """
#     label_data = []
#     for i in range(len(labels)):
#         y = np.zeros(num_classes)
#         y[int(labels[i])] = 1
#         label_data.append(y)

#     # # TODO: Fundus DR 한정 ----
#     # from datasets import  fundus_dr
#     # label_data = []
#     # for i in range(len(labels)):
#     #     y = fundus_dr._get_label(num_classes, labels[i])
#     #     label_data.append(y)

#     pos_counts = np.sum(label_data, axis=0)
#     class_pos_counts = dict(zip(class_names, pos_counts))
#     return np.shape(label_data)[0], class_pos_counts


# def get_class_weights(y, classes, severity=1):
#     #labels = np.unique(y)
#     cnt_dict = dict(Counter(y))
#     n = sum(cnt_dict.values())
#     class_weights = []
#     for label in classes:
#         class_weights.append((n/cnt_dict.get(label))**severity)
#     return class_weights
