import os
import subprocess
import sys
import time
from multiprocessing import Pool
import orjson 
import decord
import ffmpeg
from joblib import delayed, Parallel  # install psutil library to manage memory leak
from tqdm import tqdm
import pandas as pd
# from https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git install epic-kitchens library
from epic_kitchens.hoa import load_detections, DetectionRenderer
import PIL.Image
import pickle
import cv2
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List


# Define path
root_add = "/home/mona/VideoMAE/dataset/Epic_kitchen/"
save_root_add = "/home/mona/VideoMAE/dataset/Epic_kitchen/mp4_videos_BB"
# Define videos and annotations path
EPIC_100_train = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train")
EPIC_100_val = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_val")
EPIC_100_hand_objects_train = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_train")
EPIC_100_hand_objects_val = ("../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_hand_objects_val")


data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_val = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_validation.csv")


if not os.path.exists(save_root_add):
    os.makedirs(save_root_add)

if not os.path.exists(root_add + "/" + "EPIC_100_hand_objects_train_modified"):
    os.makedirs(root_add + "/" + "EPIC_100_hand_objects_train_modified")

if not os.path.exists(root_add + "/" + "EPIC_100_hand_objects_val_modified"):
    os.makedirs(root_add + "/" + "EPIC_100_hand_objects_val_modified")


#fuction for visualizing image with bounding box
def visual_bbx (images, bboxes):
    """
    images (torch.Tensor or np.array): list of images in torch or numpy type.
    bboxes (List[List]): list of list having bounding boxes in [x1, y1, x2, y2]
    """
    if isinstance(images, torch.Tensor):
        images = images.view((16, 3) + images.size()[-2:])
    color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
    if not os.path.exists('VideoMAE/scripts/data/Epic-kitchen/visual_bbx'):
        os.makedirs('VideoMAE/scripts/data/Epic-kitchen/visual_bbx')
    for i, (img, bbx) in enumerate(zip(images, bboxes)):
        if isinstance(img, Image.Image) or isinstance(img, np.ndarray):
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # (x1,y1,x2,y2) = (bbx[0], bbx[1], bbx[2], bbx[3])
        elif isinstance(img, torch.Tensor):
            frame = img.numpy().astype(np.uint8).transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # if len(bbx) != 0:
            #     (x1,y1,x2,y2) = (bbx[0][0], bbx[0][1], bbx[0][2], bbx[0][3])

        if len(bbx) != 0:
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
            ##
            for c, b in enumerate(bbx):
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color_list[0], 4)
                # cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[3]), int(b[2])), color_list[1], 4)
                # cv2.rectangle(frame, (int(b[0]), int(b[2])), (int(b[1]), int(b[3])), color_list[2], 4)
                # cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), color_list[3], 4)
                # cv2.rectangle(frame, (int(b[2]), int(b[1])), (int(b[0]), int(b[3])), color_list[4], 4)
                # cv2.rectangle(frame, (int(b[3]), int(b[1])), (int(b[2]), int(b[0])), color_list[0], 4)
                # cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[2]), int(b[3])), color_list[1], 4)
            ##
        cv2.imwrite(f'VideoMAE/scripts/data/Epic-kitchen/visual_bbx/{i}.png', frame)



def data_clean(video_root_path, idx, BB_root_path, verbose=True):

    # prepare
    video_ext = 'mp4'

    # process video & save
    start_time = time.time()

    directory = os.path.join(video_root_path, f"video_{idx}.MP4" )
    if '.' in directory.split('/')[-1]:
        video_name = directory
    else:
        video_name = '{}.{}'.format(directory, video_ext)

    try:
        # try load video

        decord_vr = decord.VideoReader(video_name, num_threads=1)

        duration = len(decord_vr)
        # if duration < 30:
        #     return [-1, -1, -1]
        video_data = decord_vr.get_batch(list(range(duration))).asnumpy()

        # get the new size (short side size 320p)
        _, img_h, img_w, _ = video_data.shape
        new_short_size = 320
        ratio = float(img_h) / float(img_w)
        if ratio >= 1.0:
            new_w = int(new_short_size)
            new_h = int(new_w * ratio / 2) * 2

        else:
            new_h = int(new_short_size)
            new_w = int(new_h / ratio / 2) * 2
        
        #scale the BBx 
        y_ratio_bb = new_h/img_h
        x_ratio_bb = new_w/img_w
        new_BB_dict = [-1, -1, -1]
        i = video_name.split('/')[-1].split('.')[0]
        BB_path = os.path.join(BB_root_path, f"detection_{idx}.pkl")
        detection = pickle.load(open(BB_path, "rb"))
        objects =  detection["objects"]
        hands = detection["hands"]
        objects_bbx = []
        objects_bbx_norm = []
        for objects_bbxs in objects:
            frame_bbx = []
            frame_bbx_norm = []
            for object_bbx in objects_bbxs:
                frame_bbx.append([object_bbx[0]*img_w*x_ratio_bb, object_bbx[1]*img_h*y_ratio_bb, object_bbx[2]*img_w*x_ratio_bb, object_bbx[3]*img_h*y_ratio_bb])
                frame_bbx_norm.append([object_bbx[0]*img_w, object_bbx[1]*img_h, object_bbx[2]*img_w, object_bbx[3]*img_h])
            objects_bbx.append(frame_bbx)
            objects_bbx_norm.append(frame_bbx_norm)

        hands_bbx = []
        hands_bbx_norm = []
        for hands_bbxs in hands:
            frame_bbx = []
            frame_bbx_norm = []
            for hand_bbx in hands_bbxs:
                frame_bbx.append([hand_bbx[0]*img_w*x_ratio_bb, hand_bbx[1]*img_h*y_ratio_bb, hand_bbx[2]*img_w*x_ratio_bb, hand_bbx[3]*img_h*y_ratio_bb])
                frame_bbx_norm.append([hand_bbx[0]*img_w, hand_bbx[1]*img_h, hand_bbx[2]*img_w, hand_bbx[3]*img_h])
            hands_bbx.append(frame_bbx)
            hands_bbx_norm.append(frame_bbx_norm)

        if "train" in BB_root_path:
            train_or_val = "train"
        elif "val" in BB_root_path:
             train_or_val = "val"
        pickle.dump({"objects":objects, "hands":hands}, open(f"{root_add}/EPIC_100_hand_objects_{train_or_val}_modified" + "/" + f"detection_{idx}.pkl", "wb"))
        new_size = (new_w, new_h)
    except Exception as e:
        # skip corrupted video files
        print("Failed to load video from {} with error {}".format(
            video_name, e))

    # visulaize video and BBs
    

    visual_bbx([video_data[0]], [hands_bbx_norm[0]])

    # process the video
    output_video_file = os.path.join(save_root_add, directory.replace('.MP4', '.mp4').split('/')[-1])

    # resize
    proc1 = (ffmpeg.input(directory).filter(
        'scale', new_size[0],
        new_size[1]).output(output_video_file).overwrite_output())
    p = subprocess.Popen(
        ['ffmpeg'] + proc1.get_args()+
        ['-hide_banner', '-loglevel', 'quiet', '-nostats'])


    end_time = time.time()
    dur_time = end_time - start_time
    if verbose:
        print(f'processing video {idx + 1} with total time {dur_time} & save video in {output_video_file}')




if __name__ == '__main__':
    # n_tasks = 64
    # new_start_idxs = [0] * n_tasks

    data_clean(EPIC_100_train, 1000, EPIC_100_hand_objects_train)
    # with Pool(n_tasks) as p:
    #     p.starmap(data_clean,[ (video_root_path, idx, BB_root_path)
    #                for (video_root_path, idx, BB_root_path) in zip ([EPIC_100_train]*len(data_train), range(len(data_train)), [EPIC_100_hand_objects_train]*len(data_train)) ])
    # with Pool(n_tasks) as p:
    #     p.starmap(data_clean,[ (video_root_path, idx, BB_root_path)
    #                for (video_root_path, idx, BB_root_path) in zip ([EPIC_100_val]*len(data_train), range(len(data_val)), [EPIC_100_hand_objects_val]*len(data_train)) ])
