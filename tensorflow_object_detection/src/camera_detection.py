import numpy as np
import tensorflow as tf
import cv2
import os

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.setUseOptimized(True)           # 

# 
###############################################
PATH_TO_CKPT = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/outputing/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'   # 
PATH_TO_LABELS = '/home/zhu/catkin_ws/src/tensorflow_object_detection/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90            # 

camera_num = 0                 # 
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


mv = cv2.VideoCapture(camera_num)  # 

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

        while True:
            ret, image_source = mv.read()  # 
            image_np = cv2.resize(image_source , (width, height), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            cv2.imshow("video", image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 
                break
