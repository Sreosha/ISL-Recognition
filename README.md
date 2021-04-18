# ISL-Recognition
Sign language is used by and for deaf and hard-hearing people. It is a gestural-visual 
language. It uses the 3-dimensional space and physical movements (Mainly handmovements), to convey meaning. It has its own unique vocabulary and syntax; different 
from spoken and written languages. A gesture can be defined as a movement, usually hand 
movements and facial movements which express idea, sentiment, emotion etc. For 
example, raising our eye brows and shrugging our shoulders are some basic gestures we 
use on regular basis. Sign language is more defined and organized way of communication 
in which every individual word is assigned to a particular gesture. With the rapid 
enhancement of technology the use of computer in one daily life has increased immensely. 
Hereby, our goal is to develop a Human Computer Interface (HCI) system which can 
understand the sign language precisely so that the signing people can communicate with the 
non- signing people with the help of an interpreter. 
Around 5% of the world community in all parts of the world is using language as a 
medium of communication. It has evolved in the course of time on a Regional-Basis. ASL 
in America, GSL in Germany, BSL in Britain and ISL in India and Pakistan.
Objectives:
Our objective is to make an efficient software by which the deaf and hard-hearing people 
all over India can communicate very easily using ISL. They can use the virtual 
Page | 7
environment like video conferencing where the people who do not know the sign language 
can also communicate with the deaf people like any other human being. Since the 
technique requires less amount of resources and expenses, common devices are very much 
compatible for the required operations it is an advantage for different people of different 
intelligence level. The people with these inabilities can express and share their words with 
the rest people of society.
Purpose, Scope and Applicability: The description of Purpose, Scope, and
Applicability are given below:
 Purpose: The purpose of our project is to provide an application by which hard￾hearing and deaf people can communicate with non-deaf people. By this 
application, a person who do not even understand Indian Sign Language (ISL) 
can communicate with the deaf or hard-hearing people without any help of a 
third person or a narrator. The aim is to create a medium between these people 
and the rest of society. This is an empowerment and lead to those people who 
hardly consider themselves as part of normal society for their inabilities.
There are mainly two motives two specific motives behind the 
development of Sign Language Recognition Model. 
The first being the development of assistive system for the deaf and hard -
hearing people. For example, Development of a natural input device for creating 
sign language documents which will make such documents much readable for 
deaf people. Moreover, hearing people have difficulties in learning sign language 
and likewise the majority of those people who were born deaf or who became 
deaf early in life, only have a limited vocabulary in community which they live. 
So, a system translating sign language into spoken language will be of immense 
help to such community.
The second is that, sign language recognition serves as a good basis for the 
interface development of gestural human-machine interface.
Page | 8  Scope: In this project, we are using our own datasets (capturing images of same 
aspect ratio using HD webcam) and that will be passed through the Google 
Mediapipe framework for the purpose of detection of the 21 points (keypoints) of a 
hand which forms a skeleton. The position of the key points vary for different 
gestures. The co-ordinate of 21 points total 42 (21 * 2) points will be calculated and 
stored in a “.txt” or “.xlsx” file. The file will be used for detecting signs by using 
LSTM model and measuring the accuracy using Machine Learning algorithms such 
as Decision Tree, Support Vector Machine (SVM), Random Forest etc. The metric 
for calculating the accuracy will be the f1_score. 
Our project is covering the interpretation part of the sign language. The sign will be 
shown with the help of the webcam of a laptop and it will detect the signs and write 
the corresponding interpretations on the screen. There are three main steps to 
perform the project: First, it will show the skeleton of hand, second it will take 
frames from the webcam and the LSTM model predict the sign, third, the machine 
learning techniques will take the dataset co-ordinate values for training and testing 
in order to determine the accuracy.
 Applicability: The technique we have used here is applicable for most of the 
virtual environments and is very much easy to use. This technique requires less 
hardware or software resources and compatible to common devices. In other 
words, to operate the software no high end devices or resources are required. We 
have used Mediapipe for detecting key points which itself a very new framework 
to the world of Artificial Intelligence and a big contribution to the projects for 
recognition and detection. Since this technique misses very few of features it 
provides result with high accuracy. 
We have not taken any available dataset from the internet. We have 
captured images of our hands (hands of team members) as the data to be processed 
by using webcam and have made our own dataset. We have created dataset for the 
letters A-Z and numbers 0-9 of Indian Sign Language (ISL). There are some signs 
which are represented through the movement of other body parts i.e not with 
hands only. For those kind of data, this technique will fail to give any output since 
we have focused only the signs that can be shown using hands only. Though this 
Page | 9
technique is applicable for any user but it needs virtual environment. That is, to 
operate the software we need devices which have camera on it otherwise no 
recognition of sign is possible.
By using our technique, people with inabilities such as hard-hearing can 
communicate with any other people who can read English in the society because 
whatever sign being recognized by the software the interpretation of the sign will 
be written on the screen.
Achievements: 
Knowledge Achievements:
In the research of this project task, we have learnt many method and technologies. These 
methods helped us finding a better one out of the other techniques and the reasons why not 
the other techniques are feasible.
We had a plan of implementing the project tasks using skin masking technique in 
which the hand portion is masked white color and the background color is black. We have 
noticed that, by using this technique, no matter what accuracy it provides in result but it 
always overlooks a huge portion of important features. In this technique, we were using 
“Convex Hull” method for recognizing the hand or the region of interest. This technique 
was able to detect the hand shape but it had faced Convexity Defects. In our case, due to 
the convexity defects the gaps between fingers could not be identified. There is a built in 
function of OpenCV library for removing the convexity defects from the frames or 
images. Even though we use that function there are still some portions which are 
uncovered and hence some features are missing or falsely interpreted. For this reason, we 
had gone for further researches, in which we got some new ideas of looking at the whole 
matter. We have seen, some researchers and developers mentioned techniques in which 
they use gloves embedded with a motion detector sensor. This technology works in very 
effective way but the problem is it needs too many resources which are normally 
unavailable to ordinary people. Another technique we have seen used by some developers 
is using key points of a hand/palm. This technique has very high accuracy but this is not a 
suitable one because it has low processing speed. In skin masking technique, the edge 
points were calculated in order to recognize the gesture which is not a good approach at
Page | 10
all. There may be some signs in which one finger rest on another. In such situations, the 
edge point will not be enough to recognize the gesture. Finally, we have taken the help of 
Mediapipe which is very much useful in detecting the keys and predicting the sign 
promptly.
Goal Achievement:
We have come up with an idea in which we are able to minimize the drawback in 
implementing our project and the result will be fruitful. May be there are some areas 
which are not being covered yet due to lack of time and resources but we have tried to 
contribute a technique which possess a good prediction.
