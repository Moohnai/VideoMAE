import pandas as pd
import csv
import os 

# create dataframe
train_df = {'path':[], 'label_name':[], 'label_num':[]}
val_df = {'path':[], 'label_name':[], 'label_num':[]}

# root addresses

root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_train.csv"
root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_validation.csv" 
video_mp4_root_add_train = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/train"
video_mp4_root_add_val = "/mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/validation"


train_label = pd.read_csv(root_add_train)
for i, item in train_label.iterrows():
    id = i
    path = os.path.join(video_mp4_root_add_train, f"video_{i}.mp4")
    if not os.path.exists(path):
        continue
    # label_name = item ['verb']
    # label_num = item ['verb_class']
    label_name = item ['noun']
    label_num = item ['noun_class']
    train_df['path'].append(path)
    train_df['label_name'].append(label_name)
    train_df['label_num'].append(label_num)
    f = open(root_add_train)

val_label = pd.read_csv(root_add_val)
for i, item in val_label.iterrows():
    id = i
    path = os.path.join(video_mp4_root_add_val, f"video_{i}.mp4")
    if not os.path.exists(path):
        continue
    # label_name = item ['verb']
    # label_num = item ['verb_class']
    label_name = item ['noun']
    label_num = item ['noun_class']
    val_df['path'].append(path)
    val_df['label_name'].append(label_name)
    val_df['label_num'].append(label_num)


train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)


# to_csv() 
csv_annotation_root = "/home/mona/VideoMAE/dataset/Epic_kitchen/annotation/noun"
if not os.path.exists(csv_annotation_root):
    os.makedirs(csv_annotation_root)
train_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "train.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)

val_df.to_csv(path_or_buf=os.path.join(csv_annotation_root, "val.csv"), sep=' ', na_rep='', float_format=None, 
columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
compression='infer', quoting=None, quotechar='"', line_terminator=None, 
chunksize=None, date_format=None, doublequote=True, escapechar=None, 
decimal='.', errors='strict', storage_options=None)