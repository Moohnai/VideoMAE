import os, glob
import pandas as pd
import csv
import decord
import cv2
import numpy as np
from decord import VideoReader
from PIL import Image
from multiprocessing import Pool

video_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS"
EPIC_100_train =  "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train"
EPIC_100_val =  "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_val"
# data_root_add = 'home/mona/VideoMAE/dataset/Epic-kitchen/raw_videos/'


data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_val = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_validation.csv")


# give a permission in terminal by: sudo chmod ugo+rwx ../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS
# if not os.path.exists(EPIC_100_train):
#     os.makedirs(EPIC_100_train)

# video_id_last = " "
# for i, item in data_train.iterrows():
#     participant_id = item["participant_id"]
#     video_id = item["video_id"]
#     start_frame = item["start_frame"]
#     stop_frame = item["stop_frame"]
#     video_path = video_root_add + "/" + f"{participant_id}" + "/videos" + "/" + f"{video_id}" + ".MP4"
    
#     # decord_vr = decord.VideoReader ("video_path")
#     if os.path.exists(f"/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train/video_{i}.MP4") == True:
#         print(f"video_{i}_train exists")
#         continue
#     else:
#         if video_id != video_id_last:
#             vr = VideoReader(video_path)
#             fps = vr.get_avg_fps()
#             # print('video frames:', len(vr))
#         extracted_frames = vr.get_batch(range (int(start_frame),  int(stop_frame)+1))

#         # save video with cv2
#         out = cv2.VideoWriter(os.path.join(EPIC_100_train, f"video_{i}.MP4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (extracted_frames.shape[2], extracted_frames.shape[1]))
#         for frame in range(extracted_frames.shape[0]):
#             frame_rgb = cv2.cvtColor(extracted_frames.asnumpy()[frame], cv2.COLOR_BGR2RGB)
#             out.write(np.array(frame_rgb))
#         out.release()
#         print(f"video_{i}.MP4 successfully saved (train)")

#         video_id_last = video_id

# dir_list = os.listdir(EPIC_100_train)
# print (f"length of train set is {len(dir_list)}")

# if not os.path.exists(EPIC_100_val):
#     os.makedirs(EPIC_100_val)

# video_id_last = " "
# for i, item in data_val.iterrows():
#     participant_id = item["participant_id"]
#     video_id = item["video_id"]
#     start_frame = item["start_frame"]
#     stop_frame = item["stop_frame"]
#     video_path = video_root_add + "/" + f"{participant_id}" + "/videos" + "/" + f"{video_id}" + ".MP4"
    
#     # decord_vr = decord.VideoReader ("video_path")
#     if os.path.exists(f"/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train/video_{i}.MP4") == True:
#         print(f"video_{i}_val exists")
#         continue
#     else:
#         if video_id != video_id_last:
#             vr = VideoReader(video_path)
#             fps = vr.get_avg_fps()
#             # print('video frames:', len(vr))
            
#         extracted_frames = vr.get_batch(range (int(start_frame),  int(stop_frame)+1))

#         # save video with cv2
#         out = cv2.VideoWriter(os.path.join(EPIC_100_val, f"video_{i}.MP4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (extracted_frames.shape[2], extracted_frames.shape[1]))
#         for frame in range(extracted_frames.shape[0]):
#             frame_rgb = cv2.cvtColor(extracted_frames.asnumpy()[frame], cv2.COLOR_BGR2RGB)
#             out.write(np.array(frame_rgb))
#         out.release()
#         print(f"video_{i}.MP4 successfully saved (val)")

#         video_id_last = video_id

# dir_list = os.listdir(EPIC_100_val)
# print (f"length of validation set is {len(dir_list)}")



#######Parallel

# give a permission in terminal by: sudo chmod ugo+rwx ../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS
if not os.path.exists(EPIC_100_train):
    os.makedirs(EPIC_100_train)


start_frames_train = []
stop_frames_train = []
video_paths_train = []
IDs_train = []
start_timestamps_train = []
stop_timestamps_train = []


for i, item in data_train.iterrows():
    IDs_train.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_train.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_train.append(stop_frame)
    video_path = video_root_add + "/" + f"{participant_id}" + "/videos" + "/" + f"{video_id}" + ".MP4"
    video_paths_train.append(video_path)
    start_timestamp = item["start_timestamp"]
    start_timestamps_train.append(start_timestamp)
    stop_timestamp = item["stop_timestamp"]
    stop_timestamps_train.append(stop_timestamp)



if not os.path.exists(EPIC_100_val):
    os.makedirs(EPIC_100_val)


start_frames_val = []
stop_frames_val = []
video_paths_val = []
IDs_val = []
start_timestamps_val = []
stop_timestamps_val = []

for i, item in data_val.iterrows():
    IDs_val.append(i)
    participant_id = item["participant_id"]
    video_id = item["video_id"]
    start_frame = item["start_frame"]
    start_frames_val.append(start_frame)
    stop_frame = item["stop_frame"]
    stop_frames_val.append(stop_frame)
    video_path = video_root_add + "/" + f"{participant_id}" + "/videos" + "/" + f"{video_id}" + ".MP4"
    video_paths_val.append(video_path)
    start_timestamp = item["start_timestamp"]
    start_timestamps_val.append(start_timestamp)
    stop_timestamp = item["stop_timestamp"]
    stop_timestamps_val.append(stop_timestamp)



EPIC_100_train = ["../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train"]*len(video_paths_train)
EPIC_100_val =  ["../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_val"]*len(video_paths_val)

def Epic_action_data_creator (start_frames, stop_frames, video_paths, EPIC_100, i):
    if 'train' in EPIC_100:
        train_or_val = 'train'
    elif 'val' in EPIC_100:
        train_or_val = 'val'

    if os.path.exists(f"/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_{train_or_val}/video_{i}.MP4") == True:
        print(f"video_{train_or_val}_{i} exists")
        return
    else:
        vr = VideoReader(video_paths)
        fps = vr.get_avg_fps()
        # print('video frames:', len(vr))
    extracted_frames = vr.get_batch(range (int(start_frames),  int(stop_frames)+1))

    # save video with cv2
    out = cv2.VideoWriter(os.path.join(EPIC_100, f"video_{i}.MP4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (extracted_frames.shape[2], extracted_frames.shape[1]))
    for frame in range(extracted_frames.shape[0]):
        frame_rgb = cv2.cvtColor(extracted_frames.asnumpy()[frame], cv2.COLOR_BGR2RGB)
        out.write(np.array(frame_rgb))
    out.release()
    print(f"video_{train_or_val}_{i}.MP4 successfully saved (train)")




# Epic_action_data_creator (start_frames_train[0],stop_frames_train[0], video_paths_train[0],EPIC_100_train[0], IDs_train[0])

n_tasks = 10 
print("Processing training videos")
with Pool(n_tasks) as p:
    p.starmap(Epic_action_data_creator,
                [(start_frames, stop_frames, video_paths, EPIC_100, i)
                for (start_frames, stop_frames, video_paths, EPIC_100, i) in zip (start_frames_train,stop_frames_train, video_paths_train,EPIC_100_train, IDs_train)  ])
print("Processing validation videos")
with Pool(n_tasks) as p:
    p.starmap(Epic_action_data_creator,
                [(start_frames, stop_frames, video_paths, EPIC_100, i)
                for (start_frames, stop_frames, video_paths, EPIC_100, i) in zip (start_frames_val,stop_frames_val, video_paths_val,EPIC_100_val, IDs_val)  ])

