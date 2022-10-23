import json, orjson
import cv2, os

with open('/home/mona/VideoMAE/SSV2_BB/bounding_box_smthsmth_scaled.json', "r", encoding="utf-8") as f:
    data = orjson.loads(f.read())

data_old={}
for i in range(1,5):
    with open(os.path.join('/home/mona/VideoMAE/SSV2_BB/',f'bounding_box_smthsmth_part{str(i)}.json'), "r", encoding="utf-8") as f:
            video_BB = orjson.loads(f.read())
    data_old.update(video_BB)

key=[]
instance_vid_len = []
for c,i in enumerate (data.keys()):
    key.append(i)
    instance_vid_len.append(len(data[i]))
    if c == 5:
        break

color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
Org_vid_len = []
frame_rate = []
for i in key:
    vid = cv2.VideoCapture(f'/home/mona/VideoMAE/dataset/somethingsomething/mp4_videos/{i}.mp4')

    fps = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    Org_vid_len.append(frame_count)
    frame_rate.append(fps)

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    # Check if camera opened successfully
    if (vid.isOpened()== False):
        print("Error opening video file")
        continue
    if not os.path.exists(f'/home/mona/VideoMAE/SSV2_BB/data/{i}/'):
        os.makedirs(f'/home/mona/VideoMAE/SSV2_BB/data/{i}/')
    if frame_count == len(data[i]):
        c = 0
        while(vid.isOpened()):
            ret, frame = vid.read()
            if ret == True:
            # Display the resulting frame    
                # cv2.imwrite(f'SSV2_BB/data/{i}/{c}.png', frame)
                # cv2.imshow('frame',frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break
                bboxs = [[x['box2d']["y1"], x['box2d']["y2"], x['box2d']["x1"], x['box2d']["x2"]] for x in data[i][c]['labels']] #y1 y2 x1 x2
                bboxs_old = [[x['box2d']["y1"], x['box2d']["y2"], x['box2d']["x1"], x['box2d']["x2"]] for x in data_old[i][c]['labels']] #y1 y2 x1 x2

                for idx, (y1,y2,x1,x2) in enumerate(bboxs):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
                for idx, (y1,y2,x1,x2) in enumerate(bboxs_old):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[1], 4)
                cv2.imwrite(f'/home/mona/VideoMAE/SSV2_BB/data/{i}/{c}.png', frame)
            else:
                break
            c += 1

        vid.release()
        cv2.destroyAllWindows()



print(instance_vid_len)   
print(Org_vid_len)





a = 0

