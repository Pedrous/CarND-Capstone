#!/usr/bin/env python
import tensorflow as tf
import cv2
import numpy as np
import os
import time

# Colors (one for each class)
#cmap = ImageColor.colormap
#COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#

def filter_boxes(class_filt, min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score` belonging to class class_filt"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score and classes[i] == class_filt:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords
"""
def draw_boxes(image, boxes, classes, thickness=4):
    #Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
"""
        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph




class TLCBBox(object):
    def __init__(self):
        #TODO load classifier
        self.input_height = 320
        self.input_width = 432
        
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        export_path = os.path.dirname(os.path.abspath(__file__)) + '/' + SSD_GRAPH_FILE
        
        self.detection_graph = load_graph(export_path)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.sess = tf.Session(config=self.config, graph=self.detection_graph)
    
        
    def get_boundingbox(self, image):
        """Determines bounding box of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            image (cv::Mat): image containing the traffic light inside a bounding box

        """
        #TODO implement light color prediction

        #img = np.array(cv2.resize(image, (self.input_width, self.input_height)))
        # Normalize
        #img = img/255.0
        
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        
        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        confidence_cutoff = 0.1
        traffic_light_class = 10
        # Filter boxes with a confidence score less than 'confidence_cutoff'
        boxes, scores, classes = filter_boxes(traffic_light_class, confidence_cutoff, boxes, scores, classes)
        
        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        #height, width = image.shape
        #width, height = draw_img.size
        #box_coords = to_image_coords(boxes, height, width)
        
        return (boxes, scores, classes)
        
        
        
        
        
        
