# Real-Time-Face-Detection-Using-Haar-Like-Features-

Haar-like features are digital image features used in object recognition. They owe their name to their intuitive similarity with Haar wavelets and were used in the first real-time face detector. We have implemented a simple real time smile detection that captures the video feed from the webcam and processes it frame by frame and makes detections on face, eyes and detects when you are you smiling.

The main advantage of the Haar like features is that they is very fast. Due to the use of integral images, a Haar-like feature of any size can be calculated in constant time.

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by 
Paul Viola and Michael Jones in their paper, [Rapid Object Detection using a Boosted Cascade of Simple Features](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.6807&rep=rep1&type=pdf) in 2001. 
It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. 
It is then used to detect objects in other images.

We have simply used haarcascade xml files to load our cascades and perform real time object detection.

# Libraries / Tools Used

* Open-cv is used for handling the video and frames.
* We have used open-cv haarcascade xml files which can be found [open-cv haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)

# Code files

* face_recog.py is a commented python script that performs detection on face ,eyes on the data coming from video feed.
* happyface_detector.py also detects when a person is smiling.


