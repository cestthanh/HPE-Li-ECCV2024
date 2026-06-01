#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 08:59:20 2024

@author: jackson-devworks
"""
import os

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from constant import experiment_config
from model import *
from utils import calulate_error, compute_pck_pckh_18
from wipose import WiPoseDataset

WIPOSE_DATASET_ROOT = os.getenv(
    "WIPOSE_DATASET_ROOT",
    "/home/research02/student1409/Wifi-HPE/data/wipose",
)
WIPOSE_PREPROCESSED_ROOT = os.getenv("WIPOSE_PREPROCESSED_ROOT")
TRAIN_BATCH_SIZE = int(os.getenv("WIPOSE_TRAIN_BATCH_SIZE", "128"))
VAL_BATCH_SIZE = int(os.getenv("WIPOSE_VAL_BATCH_SIZE", "64"))
TEST_BATCH_SIZE = int(os.getenv("WIPOSE_TEST_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.getenv("WIPOSE_NUM_WORKERS", "8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size, shuffle, drop_last=False):
    loader_kwargs = {}
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )

train_dataset, test_dataset = WiPoseDataset(
    root_dir=WIPOSE_DATASET_ROOT,
    preprocessed_root=WIPOSE_PREPROCESSED_ROOT,
), WiPoseDataset(
    root_dir=WIPOSE_DATASET_ROOT,
    split="Test",
    preprocessed_root=WIPOSE_PREPROCESSED_ROOT,
)
val_indices, test_indices = train_test_split(
    list(range(len(test_dataset))), test_size=0.5, random_state=41
)
val_data = Subset(test_dataset, val_indices)
test_data = Subset(test_dataset, test_indices)
train_loader = make_loader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = make_loader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
test_loader = make_loader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
print(
    f"wipose_root={WIPOSE_DATASET_ROOT}, preprocessed_root={WIPOSE_PREPROCESSED_ROOT}, "
    f"train_samples={len(train_dataset)}, val_samples={len(val_data)}, test_samples={len(test_data)}, "
    f"device={device}, train_batch={TRAIN_BATCH_SIZE}, val_batch={VAL_BATCH_SIZE}, workers={NUM_WORKERS}",
    flush=True,
)

torch.cuda.empty_cache()

metafi = HPEWiPoseModel().to(device)

criterion_L2 = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(metafi.parameters(), lr=0.001)
n_epochs = 20
n_epochs_decay = 30
epoch_count = 1
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: 1.0
    - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1),
)

num_epochs = experiment_config["epoch"]
pck_20_overall_max = 0
train_mean_loss_iter = []
valid_mean_loss_iter = []
time_iter = []

print(metafi._get_name() + "\n")
torch.cuda.empty_cache()
for epoch_index in tqdm(range(num_epochs)):
    loss = 0
    train_loss_iter = []
    metric = []
    metafi.train()
    relation_mean = []
    for idx, data in enumerate(train_loader):
        csi_data = data["input_wifi-csi"].to(device, non_blocking=True).float()
        keypoint = data["output"].to(device, non_blocking=True).float()

        xy_keypoint = keypoint[:, :, 0:2]
        confidence = keypoint[:, :, 2:3]

        pred_xy_keypoint, time = metafi(csi_data)  # b,2,17,17
        loss = (
            criterion_L2(
                torch.mul(confidence, pred_xy_keypoint),
                torch.mul(confidence, xy_keypoint),
            )
            / 32
        )
        train_loss_iter.append(loss.cpu().detach().numpy())
        time_iter.append(time)
        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        lr = scheduler.get_last_lr()[0]
        message = "(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) " % (
            epoch_index,
            idx * 32,
            lr,
            loss,
        )
        # print(message)
    scheduler.step()
    sum_time = np.mean(time_iter)
    train_mean_loss = np.mean(train_loss_iter)
    train_mean_loss_iter.append(train_mean_loss)

    metafi.eval()
    valid_loss_iter = []
    # metric = []
    pck_50_iter = []
    pck_20_iter = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            csi_data = data["input_wifi-csi"].to(device, non_blocking=True).float()

            keypoint = data["output"]  # 17,3
            keypoint = keypoint.to(device, non_blocking=True).float()

            xy_keypoint = keypoint[:, :, 0:2]
            confidence = keypoint[:, :, 2:3]

            pred_xy_keypoint, time = metafi(csi_data)  # 4,2,17,17
            loss = criterion_L2(
                torch.mul(confidence, pred_xy_keypoint),
                torch.mul(confidence, xy_keypoint),
            )

            valid_loss_iter.append(loss.cpu().detach().numpy())
            pred_xy_keypoint = pred_xy_keypoint.cpu()
            xy_keypoint = xy_keypoint.cpu()
            pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
            xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
            pck = compute_pck_pckh_18(
                pred_xy_keypoint_pck, xy_keypoint_pck, 0.5
            )

            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(
                compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
            )
            pck_20_iter.append(
                compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
            )

        valid_mean_loss = np.mean(valid_loss_iter)
        valid_mean_loss_iter.append(valid_mean_loss)
        mean = np.mean(metric, 0) * 1000
        mpjpe_mean = mean[0]
        pa_mpjpe_mean = mean[1]
        pck_50 = np.mean(pck_50_iter, 0)
        pck_20 = np.mean(pck_20_iter, 0)
        pck_50_overall = pck_50[18]
        pck_20_overall = pck_20[18]
        print(
            "\nvalidation result with loss: %.3f, pck_50: %.3f, pck_20: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
            % (
                valid_mean_loss,
                pck_50_overall,
                pck_20_overall,
                mpjpe_mean,
                pa_mpjpe_mean,
            )
        )

        if pck_20_overall >= pck_20_overall_max:
            print(
                "saving the model at the end of epoch %d with pck_20: %.3f"
                % (epoch_index, pck_20_overall)
            )
            checkpoint_path = os.path.join(
                experiment_config["checkpoint"], "att_wipose"
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(metafi, os.path.join(checkpoint_path, "best.pt"))
            pck_20_overall_max = pck_20_overall

        if (epoch_index + 1) % 50 == 0:
            print("the train loss for the first %.1f epoch is" % (epoch_index))
            print(train_mean_loss_iter)

epsilon = 0.4
criterion_L2 = nn.MSELoss().to(device)
loss = 0
test_loss_iter = []
metric = []
time_iter = []
pck_50_iter = []
pck_40_iter = []
pck_30_iter = []
pck_20_iter = []
pck_10_iter = []
pck_5_iter = []
metafi = torch.load(
    os.path.join(experiment_config["checkpoint"], "att_wipose", "best.pt"),
    weights_only=False,
)
metafi = metafi.to(device)
metafi.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        csi_data = data["input_wifi-csi"].to(device, non_blocking=True).float()
        # csi_dafeaturesta = csi_data.view(16,2,3,114,10)
        keypoint = data["output"]  # 17,3
        keypoint = keypoint.to(device, non_blocking=True).float()

        xy_keypoint = keypoint[:, :, 0:2]
        confidence = keypoint[:, :, 2:3]

        pred_xy_keypoint, time = metafi(csi_data)  # b,2,17,17

        loss = criterion_L2(
            torch.mul(confidence, pred_xy_keypoint),
            torch.mul(confidence, xy_keypoint),
        )
        test_loss_iter.append(loss.cpu().detach().numpy())

        pred_xy_keypoint = pred_xy_keypoint.cpu()
        xy_keypoint = xy_keypoint.cpu()
        pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
        xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)

        pck = compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)

        metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))

        pck_50_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
        )
        pck_40_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4)
        )
        pck_30_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3)
        )
        pck_20_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
        )
        pck_10_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1)
        )
        pck_5_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05)
        )

    test_mean_loss = np.mean(test_loss_iter)
    sum_time = np.sum(time_iter)
    mean = np.mean(metric, 0) * 1000
    mpjpe_mean = mean[0]
    pa_mpjpe_mean = mean[1]
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)
    pck_50_overall = pck_50[18]
    pck_40_overall = pck_40[18]
    pck_30_overall = pck_30[18]
    pck_20_overall = pck_20[18]
    pck_10_overall = pck_10[18]
    pck_5_overall = pck_5[18]
    print(
        "test result with loss: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
        % (
            test_mean_loss,
            pck_50_overall,
            pck_40_overall,
            pck_30_overall,
            pck_20_overall,
            pck_10_overall,
            pck_5_overall,
            mpjpe_mean,
            pa_mpjpe_mean,
        )
    )

    label = [
        "Nose",
        "Neck",
        "R.Shoulder",
        "R.Elbow",
        "R.Wrist",
        "L.Shoulder",
        "L.Elbow",
        "L.Wrist",
        "R.Hip",
        "R.Knee",
        "R.Ankle",
        "L.Hip",
        "L.Knee",
        "L.Ankle",
        "R.Eye",
        "L.Eye",
        "R.Ear",
        "L.Ear",
    ]

    # Tạo danh sách dữ liệu
    data = []
    for i in range(len(label)):
        data.append([label[i], pck_5[i], pck_10[i], pck_20[i], pck_30[i], pck_40[i], pck_50[i]])

    # Thêm hàng "Average"
    data.append(
        [
            "Average",
            np.mean(pck_5),
            np.mean(pck_10),
            np.mean(pck_20),
            np.mean(pck_30),
            np.mean(pck_40),
            np.mean(pck_50),
        ]
    )

    # In bảng dạng tabular
    headers = ["Keypoint", "PCK@5", "PCK@10", "PCK@20", "PCK@30", "PCK@40", "PCK@50"]
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))
