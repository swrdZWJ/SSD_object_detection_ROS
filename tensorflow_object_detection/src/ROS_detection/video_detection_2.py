# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time


import rospy
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import matplotlib
width, height = 660,440 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

start = time.time()
if cv2.setUseOptimized==True:
    print("已开启CV加速")               # 加速cv
else:
    cv2.setUseOptimized(True)



# 可能要改的内容
######################################################
PATH_TO_CKPT = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/outputing/frozen_inference_graph.pb'   # 模型及标签地址
PATH_TO_LABELS = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/data/frame_label_map.pbtxt'

video_PATH = "./testing/2017_699.mp4"              # 要检测的视频
out_PATH = "./testing/2017_001.mp4"            # 输出地址

confident = 0.5  # 置信度，即scores>confident的目标才被输出
NUM_CLASSES = 1            # 检测对象个数

fourcc = cv2.VideoWriter_fourcc(*'XVID')            # 编码器类型（可选）
# 编码器： DIVX , XVID , MJPG , X264 , WMV1 , WMV2

######################################################

#def CorpImage(img):
#    sp = img.shape#obtain the image shape
#    sz1 = sp[0]#height(rows) of image
#    sz2 = sp[1]#width(colums) of image
#    a=int(sz1/2-330) # x start
#    b=int(sz1/2+330) # x end
#    c=int(sz2/2-440) # y start
#    d=int(sz2/2+440) # y end
#    cropImg = img[a:b,c:d] #crop the image
#    return cropImg



# Load a (frozen) Tensorflow model into memory.
#detection_graph = tf.Graph()
#with detection_graph.as_default():
#  od_graph_def = tf.GraphDef()
#  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#    serialized_graph = fid.read()
#    od_graph_def.ParseFromString(serialized_graph)
#    tf.import_graph_def(od_graph_def, name='')

# Loading label map
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

# 读取视频
video_cap = cv2.VideoCapture(video_PATH)  
fps = int(video_cap.get(cv2.CAP_PROP_FPS))    # 帧率
width = int(video_cap.get(3))         # 视频长，宽
hight = int(video_cap.get(4))
videoWriter = cv2.VideoWriter(out_PATH, fourcc, fps, (width, hight)) 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True    # 减小显存占用


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



class ObjectDetection():
    def __init__(self):
       
        
        # ROS initialize
#        rospy.init_node('ros_tensorflow_ObjectDetection')
#        rospy.on_shutdown(self.shutdown)
        
        # Set model path and image topic
#        model_path = rospy.get_param("~model_path", "")
#        image_topic = rospy.get_param("~image_topic", "")
        
#        self._cv_bridge = CvBridge()
       
        
#        rospy.loginfo("finding model path...")
       
        '''select model path ,model label and model name,include 'MODEL_NAME' 'PATH_TO_CKPT' and 'PATH_TO_LABELS' '''


#        MODEL_NAME = '/outputing'
#        PATH_TO_CKPT = model_path + MODEL_NAME +'/frozen_inference_graph.pb'
#        
#        PATH_TO_LABELS = os.path.join(model_path + '/data', 'frame_label_map.pbtxt')
                
        
        # What model to download.
#        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#        MODEL_FILE = MODEL_NAME + '.tar.gz'
#        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
#        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
#        PATH_TO_LABELS = os.path.join(model_path+'/data', 'mscoco_label_map.pbtxt')
        
        
#        NUM_CLASSES = 1
        NUM_CLASSES = 90
        
        
        # Download Model
#        rospy.loginfo("Downloading models...")            #send loginfo
#        opener = urllib.request.URLopener()
#        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#        tar_file = tarfile.open(MODEL_FILE)
#        for file in tar_file.getmembers():
#            file_name = os.path.basename(file.name)       #use os.path.basename for  
#            if 'frozen_inference_graph.pb' in file_name:
#                    tar_file.extract(file, os.getcwd())   #os.getcwd() 
        
        
        #Load a (frozen) Tensorflow model into memory.
#        self.detection_graph = tf.Graph()
#        
#        
#        with self.detection_graph.as_default():
#            od_graph_def = tf.GraphDef()
#            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#                serialized_graph = fid.read()
#                od_graph_def.ParseFromString(serialized_graph)
#                tf.import_graph_def(od_graph_def, name='')
                
                
#        rospy.loginfo("loading models' label ......")
#        rospy.loginfo("please wait")
        

                
        
        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        
#        #Initialize ROS Subscriber and Publisher
#        self._sub = rospy.Subscriber(image_topic, ROSImage, self.callback, queue_size=10)    
#        self._pub = rospy.Publisher('object_detection', ROSImage, queue_size=1)
#        rospy.loginfo("Start object dectecter ...")
        
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True    #









def callback():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
                
            num = 0
            while True:
                ret, frame = video_cap.read()
                if ret == False:        # 没检测到就跳出
                    break
                num += 1
                
                
                # ROS image data to cv data
#                image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
                image_np = image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                #pil_img = Image.fromarray(cv_image)             
                #(im_width, im_height) = pil_img.size
                
                # the array based representation of the image will be used later in order to
                # prepare the result image with boxes and labels on it.
                #image_np =np.array(pil_img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                
                # Expand dimensions since the model expects images to have shape:
                #  [1, None, None, 3]
#                image_np =cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                
                #image_np = cv2.resize(cv_image , (width, height), interpolation=cv2.INTER_CUBIC)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                
                
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                
                
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                

                    
                
                
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                

                
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                
  
              
#                cv_image = self._cv_bridge.cv2_to_imgmsg(image_np, "bgr8")
                cv2.imshow("Image window", image_np)
                cv2.waitKey(1)
                
                # Publish objects image
#                ros_compressed_image=self._cv_bridge.cv2_to_imgmsg(image_np, encoding="bgr8")
#                self._pub.publish(ros_compressed_image)



if __name__ == '__main__':
    try:
        ObjectDetection()
        callback()
    except rospy.ROSInterruptException:
        rospy.loginfo("ros_tensorflow_ObjectDetection has started.")










#with detection_graph.as_default():
#  with tf.Session(graph=detection_graph, config=config) as sess:
#    num = 0
#    while True:
#        ret, frame = video_cap.read()
#        if ret == False:        # 没检测到就跳出
#            break
#        num += 1
#        print(num)  # 输出检测到第几帧了
#        # print(num/fps) # 检测到第几秒了
#        image_np = image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        #image_np = CorpImage(image_np)
#        image_np_expanded = np.expand_dims(image_np, axis=0)
#        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#        scores = detection_graph.get_tensor_by_name('detection_scores:0')
#        classes = detection_graph.get_tensor_by_name('detection_classes:0')
#        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#        # Actual detection.
#        (boxes, scores, classes, num_detections) = sess.run(
#            [boxes, scores, classes, num_detections],
#            feed_dict={image_tensor: image_np_expanded})
#
#        s_boxes = boxes[scores > confident]
#        s_classes = classes[scores > confident]
#        s_scores = scores[scores > confident]
#
#        for i in range(len(s_classes)):
#            name = i+1
#            # name = image_path.split("\\")[-1].split('.')[0]   # 不带后缀
#            ymin = s_boxes[i][0] * 660  # ymin
#            xmin = s_boxes[i][1] * 880  # xmin
#            ymax = s_boxes[i][2] * 660  # ymax
#            xmax = s_boxes[i][3] * 880  # xmax
#            score = s_scores[i]
#            if s_classes[i] in category_index.keys():
#                class_name = category_index[s_classes[i]]['name']  # 得到英文class名称
#            print("name:", name)
#            print("ymin:", ymin)
#            print("xmin:", xmin)
#            print("ymax:", ymax)
#            print("xmax:", xmax)
#            print("score:", score)
#            print("class:", class_name)
#            print("################")
#        # Visualization of the results of a detection.
#        vis_util.visualize_boxes_and_labels_on_image_array(
#            image_np,
#            np.squeeze(boxes),
#            np.squeeze(classes).astype(np.int32),
#            np.squeeze(scores),
#            category_index,
#            use_normalized_coordinates=True,
#            line_thickness=4)
#        
#        cv2.imshow("Image window", image_np)
#        cv2.waitKey(1)
#		
#        # 写视频
#        videoWriter.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#        #cv2.imshow("capture", image_np)
#        #cv2.waitKey(1)
#videoWriter.release()
#end = time.time()
#print ("Execution Time: ", end - start)
