# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:12:07 2021

@author: user
"""


import cv2
import mediapipe as mp
#from collect import get_connections_list, get_distance
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import operator


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


def get_sign_list():
    # Function to get all the values in SIGN column
    df = pd.read_csv('my_train.csv')
    return df['Sign'].unique()

def real_time_prediction():
    sign_list = get_sign_list()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections_dict = get_connections_list()

    # Initialize webcam
    # Default is zero, try changing value if it doesn't work

    hands=mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
            # Get image from webcam, change color channels and flip
        _, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        #frame=cv2.resize(frame, (400, 400))
                # Get result
        frame.flags.writeable = True
        results = hands.process(frame)
        if  results.multi_hand_landmarks:
                   
                #for hand_landmarks in results.multi_hand_landmarks:    # If hand detected, superimpose landmarks and default connections
                mp_drawing.draw_landmarks(
                        frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
                        # Get landmark coordinates & calculate length of connections
                coordinates = results.multi_hand_landmarks[0].landmark
        
                data = []
                for _, values in connections_dict.items():
                        data.append(get_distance(coordinates[values[0]], coordinates[values[1]])) 
                     
                                # Scale data
                data = np.array([data])
                data[0] /= data[0].max()
                                
                                # Load model from h5 file
                model = load_model('model_a.h5')
                
                                # Get prediction
                        #data = np.array(model(data))
                        #pred=model.evaluate(data,pred)
                        #pred = sign_list[pred.argmax()]
                data=np.reshape(data,(1,36))
                result = model.predict(data)
                prediction = {'ZERO': result[0][0], 
                        'ONE': result[0][1], 
                        'TWO': result[0][2],
                        'THREE': result[0][3],
                        'FOUR': result[0][4],
                        'FIVE': result[0][5],
                        'SIX': result[0][6], 
                        'SEVEN': result[0][7],
                        'EIGHT': result[0][8],
                        'NINE': result[0][9]}

    # Sorting based on top prediction
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
       
                frame =cv2.flip(frame, 1)
                
                                # Display text showing prediction
                cv2.putText(frame, str(prediction[0][0]), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
                
                        # Display final image
                
        
        cv2.imshow('Sign Language Detection',frame)
            # Each frame will be displayed for 20ms (50 fps)
            # Press Q on keyboard to quit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
        

if __name__ == "__main__":
    real_time_prediction()