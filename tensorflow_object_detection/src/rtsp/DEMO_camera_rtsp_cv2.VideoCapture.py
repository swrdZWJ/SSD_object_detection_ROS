import cv2

"""
2018-05-30 Yonv1943
2018-07-02 reference
Help:
XviD-1.3.5(Download size: 804 KB)
Reference: http://www.linuxfromscratch.org/blfs/view/svn/multimedia/xvid.html
Download: http://downloads.xvid.org/downloads/xvidcore-1.3.5.tar.gz
Ubuntu16.04   FFmpeg   
sudo add-apt-repository ppa:djcj/hybrid
sudo apt-get update 
sudo apt-get install ffmpeg
Can only use IE to open below website to the HIKVISION Web Camera preview page
Can not use Chrome either FireFox, even Edge cannot open it
"http://192.168.1.64/doc/page/preview.asp"
"""


def video_capture_simplify(name, pwd, ip):
    
#    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/1" % (name, pwd, ip))
    
    cap = cv2.VideoCapture("http://192.168.1.102:8080/stream_simple.html")

    while True:
        success, frame = cap.read()
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)


def video_capture(name, pwd, ip, channel_num=1):
    video_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel_num)
    window_name = "CameraIP: %s" % ip

    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    cap = cv2.VideoCapture(video_path)
    is_opened = cap.isOpened()
    print("||| CameraIP %s is opened: %s" % (ip, is_opened))

    while is_opened:
        success, frame = cap.read()  # If frame is read correctly, it will be True.
        # cap.read()  # You could use this way to skip frame

        cv2.imshow(window_name, frame) if success else None
        is_opened = False if cv2.waitKey(1) == 8 else True
        # press ENTER to quit, cv2.waitKey(1) == 13 == ord('\r')
        # press BackSpace to quit, cv2.waitKey(1) == 8 == ord('\b')

    cap.release()
    cv2.destroyWindow(window_name)


def run():
    user_name, user_pwd, camera_ip = "ubuntu", "ubuntu", "192.168.1.100"

    video_capture_simplify(user_name, user_pwd, camera_ip)
#    video_capture(user_name, user_pwd, camera_ip, channel_num=1)
#    video_capture_simplify(user_name, user_pwd, camera_ip)



if __name__ == '__main__':
    run()
    
    
    
    
    
    
