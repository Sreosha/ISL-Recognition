import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import pandas as pd

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/fortrain")
    #os.makedirs("data/fortest")
    os.makedirs("data/fortrain/0")
    os.makedirs("data/fortrain/1")
    os.makedirs("data/fortrain/2")
    os.makedirs("data/fortrain/3")
    os.makedirs("data/fortrain/4")
    os.makedirs("data/fortrain/5")
    os.makedirs("data/fortrain/6")
    os.makedirs("data/fortrain/7")
    os.makedirs("data/fortrain/8")
    os.makedirs("data/fortrain/9")
    #os.makedirs("data/test/0")
    #os.makedirs("data/test/1")
    #os.makedirs("data/test/2")
    #os.makedirs("data/test/3")
    #os.makedirs("data/test/4")
    #os.makedirs("data/test/5")
    #os.makedirs("data/test/6")
    #os.makedirs("data/test/7")
    #os.makedirs("data/test/8")
    #os.makedirs("data/test/9")
    
mode = 'train'
directory = 'data/'+'fortrain'+'/'
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



def get_connections_list():
    # All landmark names and values: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
    return {
        "WRIST_TO_THUMB_MCP": (0, 2),
        "WRIST_TO_THUMB_IP": (0, 3),
        "WRIST_TO_THUMB_TIP": (0, 4),
        "WRIST_TO_INDEX_FINGER_PIP": (0, 6),
        "WRIST_TO_INDEX_FINGER_DIP": (0, 7),
        "WRIST_TO_INDEX_FINGER_TIP": (0, 8),
        "WRIST_TO_MIDDLE_FINGER_PIP": (0, 10),
        "WRIST_TO_MIDDLE_FINGER_DIP": (0, 11),
        "WRIST_TO_MIDDLE_FINGER_TIP": (0, 12),
        "WRIST_TO_RING_FINGER_PIP": (0, 14),
        "WRIST_TO_RING_FINGER_DIP": (0, 15),
        "WRIST_TO_RING_FINGER_TIP": (0, 16),
        "WRIST_TO_PINKY_PIP": (0, 18),
        "WRIST_TO_PINKY_DIP": (0, 19),
        "WRIST_TO_PINKY_TIP": (0, 20),
        "THUMB_MCP_TO_THUMB_TIP": (2, 4),
        "INDEX_FINGER_MCP_TO_INDEX_FINGER_TIP": (5, 8),
        "MIDDLE_FINGER_MCP_TO_MIDDLE_FINGER_TIP": (9, 12),
        "RING_FINGER_MCP_TO_RING_FINGER_TIP": (13, 16),
        "PINKY_MCP_TO_PINKY_TIP": (17, 20),
        "THUMB_TIP_TO_INDEX_FINGER_MCP": (4, 5),
        "THUMB_TIP_TO_INDEX_FINGER_PIP": (4, 6),
        "THUMB_TIP_TO_INDEX_FINGER_DIP": (4, 7),
        "THUMB_TIP_TO_INDEX_FINGER_TIP": (4, 8),
        "THUMB_TIP_TO_MIDDLE_FINGER_MCP": (4, 9),
        "THUMB_TIP_TO_MIDDLE_FINGER_PIP": (4, 10),
        "THUMB_TIP_TO_MIDDLE_FINGER_DIP": (4, 11),
        "THUMB_TIP_TO_MIDDLE_FINGER_TIP": (4, 12),
        "THUMB_TIP_TO_RING_FINGER_MCP": (4, 13),
        "THUMB_TIP_TO_RING_FINGER_PIP": (4, 14),
        "THUMB_TIP_TO_RING_FINGER_DIP": (4, 15),
        "THUMB_TIP_TO_RING_FINGER_TIP": (4, 16),
        "THUMB_TIP_TO_PINKY_MCP": (4, 17),
        "THUMB_TIP_TO_PINKY_PIP": (4, 18),
        "THUMB_TIP_TO_PINKY_DIP": (4, 19),
        "THUMB_TIP_TO_PINKY_TIP": (4, 20)
    }

def get_distance(first, second):
    # Calculate distance from two coordinates
    return np.sqrt(
        (first.x - second.x) ** 2 
        + (first.y - second.y) ** 2 
        + (first.z - second.z) ** 2
    )


 # Simulating mirror image

#data0=[];data1=[];data0=[];data2=[];data3=[];data4=[];data5=[];data6=[];data7=[];data8=[];data9=[]
data=[]
connections_dict= get_connections_list()
hands = mp_hands.Hands(
    min_detection_confidence=0.5,max_num_hands=1, min_tracking_confidence=0.5)
while cap.isOpened():
    row0=[];row1=[];row2=[];row3=[];row3=[];row4=[];row5=[];row6=[];row7=[];row8=[];row9=[]
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (600, 600)) 
    h, w, _ = frame.shape
    #frame= frame[0:h, 0:w]

    # Getting count of existing images
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six': len(os.listdir(directory+"/6")),
             'seven': len(os.listdir(directory+"/7")),
             'eight': len(os.listdir(directory+"/8")),
             'nine': len(os.listdir(directory+"/9"))}
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SIX : "+str(count['six']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "EIGHT : "+str(count['eight']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "NINE : "+str(count['nine']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    if (results.multi_hand_landmarks):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        coordinates = results.multi_hand_landmarks[0].landmark
   # def cord():
        #for _,values in connections_dict.items():
           # first,second=coordinates[values[0]], coordinates[values[1]]
           # get_distance(first, second)
    cv2.imshow('Hands', frame)
    
   
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row0.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row0.append(0)
        data.append(list(row0))
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row1.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row1.append(1)
        data.append(row1)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row2.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row2.append(2)
        data.append(row2)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row3.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row3.append(3)
        data.append(row3)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row4.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row4.append(4)
        data.append(row4)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row5.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row5.append(5)
        data.append(row5)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row6.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row6.append(6)
        data.append(row6) 
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row7.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row7.append(7)
        data.append(row7)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row8.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row8.append(8)
        data.append(row8)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', frame)
        for _,values in connections_dict.items():
            row9.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
        row9.append(9)
        data.append(row9)
        
cap.release()
cv2.destroyAllWindows()

cols=list(connections_dict.keys())
cols.append('Sign')

df=pd.DataFrame(data=data,columns=cols)
df.to_csv('my_train.csv')
