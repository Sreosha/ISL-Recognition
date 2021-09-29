# INDIAN SIGN LANGUAGE RECOGNITION
This project recognizes the signs between 0-9 in Indian Sign Language (ISL). The main goal of this project is to help the deaf and hard hearing people.In order to use the code you may download the zip or clone the link.
# DESCRIPTION
First I have collected my own data for this project. I have used Google Mediapipe for Hands in order to collect the data features (21 3D keypoints of a palm). Second, I have calculated the distances between the keypoints of a hand pose (for example, distance between index tip to index dip) using Eucledian distance which considered as the features for a pose and labeled the sign. Then these features, along with their labels are saved in the csv file. 

This is how my csv dataset looks like for pose 0
![image](https://user-images.githubusercontent.com/63660066/135231301-63993e41-ab3b-49e0-89dd-26e53c755c39.png)

These are the 10 poses (The pictures are in order) I have used


![image](https://user-images.githubusercontent.com/63660066/135232141-200ec348-cdf2-4df1-a423-4e11b0a6a042.png)
![image](https://user-images.githubusercontent.com/63660066/135232235-1b64e1ae-ad4c-4d8e-af5a-e4bf3da1cc55.png)
![image](https://user-images.githubusercontent.com/63660066/135232374-2e4b20b2-6d56-42e8-a0c4-c92613b82ef5.png)
![image](https://user-images.githubusercontent.com/63660066/135232600-578e06ed-39fa-4ec8-96a1-e683f38aa682.png)
![image](https://user-images.githubusercontent.com/63660066/135232734-6cd65973-c5df-4e5c-86ff-c173b94e068d.png)
![image](https://user-images.githubusercontent.com/63660066/135233040-f4869f06-b8f7-4861-bf9a-3cf001f04b26.png)
![image](https://user-images.githubusercontent.com/63660066/135233173-45bb53b5-ec83-40df-a2b0-a7149c571566.png)
![image](https://user-images.githubusercontent.com/63660066/135233423-d9a331d8-cb67-4863-bdfb-d106a4124114.png)
![image](https://user-images.githubusercontent.com/63660066/135233585-8cf4e45f-828c-4fc1-b350-4ea3333094ca.png)
![image](https://user-images.githubusercontent.com/63660066/135233709-be079b10-098f-4a8b-aa6b-0b56d5cd7fda.png)




Follow the link for learning about Google Mediapipe:
https://google.github.io/mediapipe/solutions/hands.html#:~:text=MediaPipe%20Hands%20is%20a%20high,from%20just%20a%20single%20frame.&text=Tracked%203D%20hand%20landmarks%20are,landmarks%20closer%20to%20the%20camera.
# COLLECTION OF DATA
You can collect 10 classes (0-9) of data. In order to collect the data you have to run 'collect.py' file. After running the file, you will see the webcam turned on. Next you have to pose any sign between 0-9 and hit the corresding sign from keyboard. For example, for collecting data of pose 1, you have to do the following steps-
  i) Pose for sign 1
  ii) Hit 1 from the keyboard while posing
Do the same for all the signs and then hit on esc to turn off the webcam and stop the collection of data. I will recommend to take at least 1000 data for each signs. 
The features are the saved in the form of CSV file.
If you want to work with my data then you can simply skip the collection and training steps and run the 'predict.py'.
# TRAINING
Just by running the 'train.py' you can train your data and save the model in the form .h5/.hdf5 file.
# PREDICTION
In order to do the prediction on real time frames, run the 'predict.py'.
