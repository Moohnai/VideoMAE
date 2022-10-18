import json
import cv2, os

f = open('/home/mona/SSV2_BB/annt.json')
data = json.load(f)

key=[]
instance_vid_len = []
for i in data.keys():
    key.append(i)
    instance_vid_len.append(len(data[i]))

color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
Org_vid_len = []
frame_rate = []
for i in key:
    vid = cv2.VideoCapture(f'/home/mona/VideoMAE/dataset/somethingsomething/20bn-something-something-v2/{i}.webm')

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
    if not os.path.exists(f'SSV2_BB/data/{i}/'):
        os.makedirs(f'SSV2_BB/data/{i}/')
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
                bboxs = [list(x['box2d'].values()) for x in data[i][c]['labels']] #y1 y2 x1 x2

                if i == '2003':
                    a=0

                for idx, (y1,y2,x1,x2) in enumerate(bboxs):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[idx], 4)
                cv2.imwrite(f'SSV2_BB/data/{i}/{c}.png', frame)
            else:
                break
            c += 1

        vid.release()
        cv2.destroyAllWindows()



print(instance_vid_len)   
print(Org_vid_len)





a = 0

