import csv
import glob
import os
from multiprocessing import Pool
import cv2
import decord
import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image
import PIL.Image
from epic_kitchens.hoa import load_detections


# Define videos and annotations path
video_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS"
annot_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/annotation/hand-objects"

EPIC_100_train = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train_new")
EPIC_100_val = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_val")
EPIC_100_hand_objects_train = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_train")
EPIC_100_hand_objects_val = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_val")



data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train_fps.csv")
data_val = pd.read_csv(
    "VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_val_fps.csv"
)


start_frames_train = []
stop_frames_train = []
video_paths_train = []
IDs_train = []
fps_train = []
annot_paths_train = []
video_ids_train = []
for i, item in data_train.iterrows():
    IDs_train.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    video_ids_train.append(video_id)
    start_frame = item["start_frame"]
    start_frames_train.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_train.append(stop_frame)
    fps_train.append(item["fps"])
    video_path = (
        video_root_add
        + "/"
        + f"{participant_id}"
        + "/rgb_frames"
        + "/"
        + f"{video_id}"
    )
    video_paths_train.append(video_path)
    annot_path = (
        annot_root_add
        + "/"
        + f"{participant_id}"
        + "/"
        + f"{video_id}"
        + ".pkl"
    )
    annot_paths_train.append(annot_path)




start_frames_val = []
stop_frames_val = []
video_paths_val = []
IDs_val = []
fps_val = []
annot_paths_val = []
video_ids_val = []
for i, item in data_val.iterrows():
    IDs_val.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    video_ids_val.append(video_id)
    start_frame = item["start_frame"]
    start_frames_val.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_val.append(stop_frame)
    fps_val.append(item["fps"])
    video_path = (
        video_root_add
        + "/"
        + f"{participant_id}"
        + "/rgb_frames"
        + "/"
        + f"{video_id}"
    )
    video_paths_val.append(video_path)
    annot_path = (
        annot_root_add
        + "/"
        + f"{participant_id}"
        + "/"
        + f"{video_id}"
        + ".pkl"
    )
    annot_paths_val.append(annot_path)

    

def check_video_annot(video_path, annot_path, video_id):
    v_id = "P00"
    for i in range(len(video_path)):
        if video_id[i] != v_id:
            video = glob.glob(video_path[i] + "/*.jpg")
            detection = load_detections(annot_path[i])
            if len(video) != len(detection):
                print(f"for video {video_id[i]} and corresponding annotation we have different frames!!!!!!!")
            else:
              print(f"{video_id[i]} : everything is fine")
            v_id = video_id[i]


# i = 0
# check_video_annot(video_paths_train[i], annot_paths_train[i], video_ids_train[i])

print("checking for training videos...")
check_video_annot(video_paths_train, annot_paths_train, video_ids_train)
print("checking for val videos...")
check_video_annot(video_paths_val, annot_paths_val, video_ids_val)

