import numpy as np
import tensorflow as tf
import cv2
import os
#import PIL


from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setUseOptimized(True)           # 

# 
###############################################
PATH_TO_CKPT = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/outputing/ssd_mobilenet_v2_coco/frozen_inference_graph.pb'   # 
PATH_TO_LABELS = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90            # 

camera_num = 0                 # 
cam_url='http://192.168.1.102:8080/?action=stream'

confident = 0.7

width, height = 640,480    # 
###############################################

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#mv = cv2.VideoCapture(camera_num)  # 

mv = cv2.VideoCapture(cam_url)

fps = int(mv.get(cv2.CAP_PROP_FPS))


mv.set(3, width)     # 
mv.set(4, height)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True




with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
#        number = 0
        
        
        
        while True:
            ret, image_source = mv.read()  #
            
            if ret == False: 
                break    
            
#            number += 1
                
            
#            print(number)
#            print(image_source.shape)
#            print(fps)
            
            
            image_np = cv2.resize(image_source , (width, height), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            
            
            
            s_boxes = boxes[scores > confident]
            s_classes = classes[scores > confident]
            s_scores = scores[scores > confident]
            
            for i in range(len(s_classes)):
                
                name = i+1
            # name = image_path.split("\\")[-1].split('.')[0]   # 
                ymin = s_boxes[i][0] * height  # ymin
                xmin = s_boxes[i][1] * width  # xmin
                ymax = s_boxes[i][2] * height  # ymax
                xmax = s_boxes[i][3] * width  # xmax
                score = s_scores[i]
                if s_classes[i] in category_index.keys():
                    class_name = category_index[s_classes[i]]['name']  # 
                print("name:", name)
                print("ymin:", ymin)
                print("xmin:", xmin)
                print("ymax:", ymax)
                print("xmax:", xmax)
                print("score:", score)
                print("class:", class_name)
                print("################")
            
            
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            
            
            cv2.imshow("video_object_detection", image_np)
#            cv2.imshow("video_real", image_source)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 
                break
