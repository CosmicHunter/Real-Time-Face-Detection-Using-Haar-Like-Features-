import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_happy_face(grayscale_image,original_image):
    face_rect = face_cascade.detectMultiScale(grayscale_image , scaleFactor = 1.3 ,minNeighbors= 5)
    
    for (x,y,w,h) in face_rect:
        cv2.rectangle(original_image,(x,y),(x+w,y+h),(255,0,0),2)
        region_of_interest_grayimg = grayscale_image[y : y+h , x : x+w]
        region_of_interest_colorimg = original_image[y:y+h , x:x+w]
        # Cascades are applied in gray images
        eye_rectangle = eye_cascade.detectMultiScale(region_of_interest_grayimg , scaleFactor = 1.1 , minNeighbors = 20)
        # We will always draw rectangle on the original part of the image i.e colored image
        for (eye_x,eye_y,eye_width,eye_height) in eye_rectangle:
            cv2.rectangle(region_of_interest_colorimg , (eye_x,eye_y),(eye_x + eye_width,eye_height+eye_y),(0,255,0),2)
        smile_rect = smile_cascade.detectMultiScale(region_of_interest_grayimg,scaleFactor = 1.6, minNeighbors = 22)
        # The number of neighbours we have increased in order to keep the threshold more for detecting a smile
        # if the min neighbours are less than many things that look like smiling get detected.
        # In other words many red boxes appear
        for (sx,sy,sw,sh) in smile_rect:
            cv2.rectangle(region_of_interest_colorimg,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return original_image

video_capture_obj = cv2.VideoCapture(0)
count = 0
while True:
    _,img = video_capture_obj.read()
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    detected_canvas = detect_happy_face(gray_img,img)
    cv2.imshow("Video",detected_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKey(1) displays the image for 1 milli seconds
       break
    
    if cv2.waitKey(1) & 0xFF == ord('c'):    # Press 'c' key to save the image frame
           count = count + 1
           crop_img = detected_canvas[100:600, 100:400] # Crop from x, y, w, h -> 100, 200, 300, 400
           cv2.imwrite("HappyFaceImage"+str(count)+".jpg", crop_img)

video_capture_obj.release()
cv2.destroyAllWindows()
