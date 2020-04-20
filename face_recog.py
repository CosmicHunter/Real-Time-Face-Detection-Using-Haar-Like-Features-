

import cv2

# Loading the Cascades 
"""
We have two cascade xml files in the folder. There is one cascade file for the face and one cascade for the eyes.
First step is to load the cascades one for the face and one for the eye.
We have cascade Classifier class in open cv . We will use that class 

We will use the cv::CascadeClassifier class to detect objects in a video stream. Particularly, we will use the functions:

    cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP classifier
    cv::CascadeClassifier::detectMultiScale to perform the detection.



Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by 
Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. 
It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. 
It is then used to detect objects in other images.
"""

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

"""
Now we will implement a function that will use these cascade objects to detect the faces , eyes
.
"""

# This function will take the single images from the video stream and do the detection.
def detect_target(grayscale_image , original_image):
    face_rectangle = face_cascade.detectMultiScale(grayscale_image,scaleFactor = 1.3 , minNeighbors = 5)                 
    
     
    # CascadeClassifier :: detectMultiScale()  --> Detects objects of different sizes in the input image. The detected 
    # objects are returned as a list of rectangles.
    # First we will get  the cordinates of the faces
    
#     Parameters:	
# =============================================================================
#     cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
#     image – Matrix of the type CV_8U containing an image where objects are detected.
#     objects – Vector of rectangles where each rectangle contains the detected object.
#     scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
#     minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
#     flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
#     minSize – Minimum possible object size. Objects smaller than that are ignored.
#     maxSize – Maximum possible object size. Objects larger than that are ignored.
# 
# =============================================================================
    """ Now we will iterate on the list of rectangles returned by the detectMultiscale function and draw rectangles on it
        cv2.rectangle(image, start_point, end_point, color, thickness) To draw rectangle we will use this function.
# =============================================================================
#         start_point: It is the starting coordinates of rectangle. The coordinates are represented as tuples of 
                       two values i.e. (X coordinate value, Y coordinate value).
          
          end_point:  It is the ending coordinates of rectangle. The coordinates are represented as tuples of two 
                      values i.e. (X coordinate value, Y coordinate value).
# =============================================================================
    """
    
    """
    Important thing to note is once we have the face rectangle , So obviously eyes will also be within those rectangles
    Hence we will detect eyes in the reference frame of the face rectangle to save computation. 
    Also note that image is a matrix and its row is y dimension of the rectanglae and x is columns.
    """
    
    for (x,y,width,height) in face_rectangle:
        cv2.rectangle(original_image,(x,y),(x+width , y+height) ,(255,0,0),2)
        region_of_interest_grayimg = grayscale_image[y : y+height , x : x+width ]
        region_of_interest_colorimg = original_image[y:y+height , x:x+width]
        # Cascades are applied in gray images
        eye_rectangle = eye_cascade.detectMultiScale(region_of_interest_grayimg , scaleFactor = 1.1 , minNeighbors = 3)
        # We will always draw rectangle on the original part of the image i.e colored image
        for (eye_x,eye_y,eye_width,eye_height) in eye_rectangle:
            cv2.rectangle(region_of_interest_colorimg , (eye_x,eye_y),(eye_x + eye_width,eye_height+eye_y),(0,0,255),2)
    return original_image


"""
# =============================================================================
#   Now we will use the video stream from the webcam and on each of the images coming from the Webcam and on each of the
    images we will apply the detect_target() function to detect faces and eyes. 
    
    
    We will use the Video Capture Class
    Class for video capturing from video files, image sequences or cameras
    
    cv2.VideoCapture() →  it returns ====>>  <VideoCapture object>
    
# =============================================================================

"""

# camera_port = 0 for computer Webcam
# camera_port = 1 for external Webcam
video_capture_object = cv2.VideoCapture(0)        

count = 0
while True:
    _,image = video_capture_object.read()
    gray_img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    detected_canvas = detect_target(gray_img,image)
    cv2.imshow("Video",detected_canvas)   # Video is window name
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKey(1) displays the image for 1 milli seconds
       break
    
    if cv2.waitKey(1) & 0xFF == ord('c'):   # Press 'c' to save image frame
           count+=1
           crop_img = detected_canvas[100:600, 100:400] # Crop from x, y, w, h -> 100, 200, 300, 400
           cv2.imwrite("face"+str(count)+".jpg", crop_img)
    
    """
# =============================================================================
#     cv2.VideoCapture.read([image]) → retval, image
      Grabs, decodes and returns the next video frame.
      
      Python: cv2.VideoCapture.release() → None
      Closes video file and capturing device
      
      
      
      1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
      2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
      
      
      
      
      

    ord('q') returns the Unicode code point of q
    cv2.waitkey(1) returns a 32-bit integer corresponding to the pressed key
    & 0xFF is a bit mask which sets the left 24 bits to zero, because ord() returns a value betwen 0 and 255, since your keyboard only has a limited character set
    Therefore, once the mask is applied, it is then possible to check if it is the corresponding key.


# =============================================================================
   """
   # TO stop the webcam and face detection 
   
video_capture_object.release()
cv2.destroyAllWindows() #  Destroys all of the HighGUI windows.     
        