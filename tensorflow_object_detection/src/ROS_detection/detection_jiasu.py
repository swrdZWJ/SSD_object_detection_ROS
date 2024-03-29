#!/usr/bin/env python




#ROSLAUNCH :
#<param name="image_topic" value="/usb_cam/image_raw" />
#<param name="model_path" value="$(find tensorflow_object_detection)/object_detection" />





import rospy
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import matplotlib
import numpy as np

#import time

import os
#import six.moves.urllib as urllib  
#import sys
#import tarfile

import tensorflow as tf
#import zipfile

width, height = 660,440    
#from collections import defaultdict
#from io import StringIO
  
#from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setUseOptimized(True)           # speed cv


#Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class ObjectDetection():
    def __init__(self):
       
        
        # ROS initialize
        rospy.init_node('ros_tensorflow_ObjectDetection')
        rospy.on_shutdown(self.shutdown)
        
        # Set model path and image topic
        model_path = rospy.get_param("~model_path", "")
        image_topic = rospy.get_param("~image_topic", "")
        
        self._cv_bridge = CvBridge()
       
        
        rospy.loginfo("finding model path...")
       
        '''select model path ,model label and model name,include 'MODEL_NAME' 'PATH_TO_CKPT' and 'PATH_TO_LABELS' '''


#        MODEL_NAME = '/outputing/ssd_mobilenet_v1_coco'
        MODEL_NAME = '/outputing/ssd_mobilenet_v2_coco'
        
        PATH_TO_CKPT = model_path + MODEL_NAME +'/frozen_inference_graph.pb'
        
        PATH_TO_LABELS = os.path.join(model_path + '/data', 'mscoco_label_map.pbtxt')
                
        
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
        self.detection_graph = tf.Graph()
        
        
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
                
        rospy.loginfo("loading models' label ......")
        rospy.loginfo("please wait")
        

                
        
        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        
        #Initialize ROS Subscriber and Publisher
        self._sub = rospy.Subscriber(image_topic, ROSImage, self.callback, queue_size=20)    
#        self._pub = rospy.Publisher('object_detection', ROSImage, queue_size=1)
        
        
        rospy.loginfo("Start object dectecter ...")
        
        
        
    def callback(self, image_msg):
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        rospy.loginfo("start detection!")
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                
                # ROS image data to cv data
                image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
                
                rospy.loginfo("transport finished!")
                #pil_img = Image.fromarray(cv_image)             
                #(im_width, im_height) = pil_img.size
                
                # the array based representation of the image will be used later in order to
                # prepare the result image with boxes and labels on it.
                #image_np =np.array(pil_img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                
                # Expand dimensions since the model expects images to have shape:
                #  [1, None, None, 3]
                #image_np = image_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
                
                
                #image_np = cv2.resize(cv_image , (width, height), interpolation=cv2.INTER_CUBIC)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                
                rospy.loginfo("1")
                
                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                
                rospy.loginfo("2")
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                
                rospy.loginfo("3")
                
                
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                rospy.loginfo("4")
                
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                
                
                rospy.loginfo("5")
  
              
#                cv_image = self._cv_bridge.cv2_to_imgmsg(image_np, "bgr8")
                cv2.imshow("Image window", image_np)
                cv2.waitKey(1)
                
                rospy.loginfo("6")
                
                # Publish objects image
#                ros_compressed_image=self._cv_bridge.cv2_to_imgmsg(image_np, encoding="bgr8")
#                self._pub.publish(ros_compressed_image)
                
    def shutdown(self):
        rospy.loginfo("Stopping the tensorflow object detection...")
        rospy.sleep(1)
        
     
        
        
if __name__ == '__main__':
    try:
        ObjectDetection()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ros_tensorflow_ObjectDetection has started.")
        
        
        


























