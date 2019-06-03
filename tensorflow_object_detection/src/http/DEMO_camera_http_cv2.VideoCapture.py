
import cv2
cv2.namedWindow("webcam test")
cam_url='http://192.168.1.102:8080/?action=stream'
vc=cv2.VideoCapture(cam_url)
if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False
    print('0')

while rval:
#    frame=cv2.resize(frame,(100,100)) 
    cv2.imshow("webcam test", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

#
