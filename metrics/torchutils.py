# 数据增强和测试指标的代码集中在这里
# 导入必备的包
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math
# 网络模型构建需要的包
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
# Metric 测试准确率需要的包
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,confusion_matrix
# Augmentation 数据增强要使用到的包
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets, models, transforms

# 这个库主要用于定义如何进行数据增强。
# https://zhuanlan.zhihu.com/p/149649900?from_voters_page=true
def get_torch_transforms(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.Resize(256),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


# 训练集的预处理以及数据增强
def get_train_transforms(img_size=320):
    return albumentations.Compose(
        [
            albumentations.Resize(img_size, img_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            # albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            # albumentations.Random
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0),
        ]
    )


# 验证集和测试集的预处理
def get_valid_transforms(img_size=224):
    return albumentations.Compose(
        [
            albumentations.Resize(img_size, img_size),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )


# 加载csv格式的数据
class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.images_filepaths = images_filepaths  # 数据集路径是个列表
        self.labels = labels  # 标签也是个列表
        self.transform = transform  # 数据增强

    def __len__(self):
        # 返回数据的长度
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        # 迭代器，这里使用的是cv，所以一定不要出现中文路径
        image_filepath = self.images_filepaths[idx]
        # print(image_filepath)
        image = cv2.imread(image_filepath)  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 图片的颜色通道转化
        label = self.labels[idx]  # 读取图片标签
        if self.transform is not None:  # 对图片做处理
            image = self.transform(image=image)["image"]  # 懂了！这个转化做的是传入一个图片，返回的是一个字典，我们应该将转化之后的图片那部分取出
        # 返回处理之后的图片和标签
        return image, label  #


# 测试准确率
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# 计算f1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    # tp fn fp
    return recall_score(target, y_pred, average="macro", zero_division=0)

# 计算precision
def calculate_precision_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return precision_score(target, y_pred, average='macro')
# 计算precision

def specificity_score(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Specificity = TN / (TN + FP)
    # For binary classification, cm[0, 0] is TN and cm[0, 1] is FP
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-15)  # Adding a small epsilon to avoid division by zero

    return specificity


def calculate_specificity_macro(output, target):
    # Convert output to probabilities and then to predicted labels
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()

    # Convert target to cpu tensor
    target = target.cpu()

    # Calculate specificity for each class
    num_classes = len(torch.unique(target))
    total_specificity = 0.0
    for class_idx in range(num_classes):
        # Create binary targets for specific class
        class_target = (target == class_idx).long()
        class_pred = (y_pred == class_idx).long()

        # Calculate specificity for current class
        class_specificity = specificity_score(class_target, class_pred)
        total_specificity += class_specificity

    # Macro-average specificity
    macro_specificity = total_specificity / num_classes

    return macro_specificity


# 训练的时候输出信息使用
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# 调整学习率
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


""" learning rate schedule """


# 计算学习率
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr
