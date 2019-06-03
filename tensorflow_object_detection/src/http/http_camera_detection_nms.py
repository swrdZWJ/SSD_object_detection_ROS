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


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        picked_boxes = np.zeros((1,4)) 
        picked_score = np.zeros(1)
        return picked_boxes, picked_score

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes, dtype = float), np.array(picked_score, dtype = float) 




# Draw parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

# IoU threshold
threshold = 0.4






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
            
            
#            print(boxes.shape)
#            picked_boxes, picked_score = nms(boxes, scores, threshold)            
            
            s_boxes = boxes[scores > confident]
#            print(-------------)
#            print(scores)
#            print(s_boxes)
#            print(***********)
            s_classes = classes[scores > confident]
            s_scores = scores[scores > confident]
            
            
            
            picked_boxes, picked_score = nms(s_boxes, s_scores, threshold)
            
            picked_size = len(picked_score)
            ss_boxes = np.row_stack((picked_boxes, np.zeros((100-picked_size,4))))
            ss_scores = np.append(picked_score, np.zeros(100-picked_size))
            #s_classes = classes[scores == picked_score]
            picked_boxes_expanded = np.expand_dims(ss_boxes, axis=0)
            picked_scores_expanded = np.expand_dims(ss_scores, axis=0)
            
#            for i in range(len(s_classes)):
#                
#                name = i+1
#            # name = image_path.split("\\")[-1].split('.')[0]   # 
#                ymin = s_boxes[i][0] * height  # ymin
#                xmin = s_boxes[i][1] * width  # xmin
#                ymax = s_boxes[i][2] * height  # ymax
#                xmax = s_boxes[i][3] * width  # xmax
#                score = s_scores[i]
#                if s_classes[i] in category_index.keys():
#                    class_name = category_index[s_classes[i]]['name']  # 
#                print("name:", name)
#                print("ymin:", ymin)
#                print("xmin:", xmin)
#                print("ymax:", ymax)
#                print("xmax:", xmax)
#                print("score:", score)
#                print("class:", class_name)
#                print("################")
            
            
            
#            vis_util.visualize_boxes_and_labels_on_image_array(
#                image_np,
#                np.squeeze(boxes),
#                np.squeeze(classes).astype(np.int32),
#                np.squeeze(scores),
#                category_index,
#                use_normalized_coordinates=True,
#                line_thickness=4)
            
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(picked_boxes_expanded),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(picked_scores_expanded),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            
            
            cv2.imshow("video_object_detection", image_np)
#            cv2.imshow("video_real", image_source)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 
                break
