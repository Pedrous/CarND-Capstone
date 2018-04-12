#!/usr/bin/env python
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import os
import time
from tl_ssd import TLCBBox

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[0] = boxes[0] * height
    box_coords[1] = boxes[1] * width
    box_coords[2] = boxes[2] * height
    box_coords[3] = boxes[3] * width
    
    return box_coords



class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.input_height = 320
        self.input_width = 432
        
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=self.config)
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        img_key = 'img'
        kp_key = 'kp'
        pred_key = 'pred'
        logits_key = 'logits'
        
        export_path = os.path.dirname(os.path.abspath(__file__)) + '/model'
        #export_path = '/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/model'

        meta_graph_def = tf.saved_model.loader.load(
                            self.sess,
                            [tf.saved_model.tag_constants.SERVING],
                            export_path)
        signature = meta_graph_def.signature_def

        img_tensor_name = signature[signature_key].inputs[img_key].name
        kp_tensor_name = signature[signature_key].inputs[kp_key].name
        pred_tensor_name = signature[signature_key].outputs[pred_key].name
        logits_tensor_name = signature[signature_key].outputs[logits_key].name

        self.img = self.sess.graph.get_tensor_by_name(img_tensor_name)
        self.keep_prob = self.sess.graph.get_tensor_by_name(kp_tensor_name)
        self.pred = self.sess.graph.get_tensor_by_name(pred_tensor_name)
        self.logits = self.sess.graph.get_tensor_by_name(logits_tensor_name)
        
        self.light_boundingboxer = TLCBBox()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # Get the bounding box
        boxes, scores, classes = self.light_boundingboxer.get_boundingbox(image)
        #print scores
        try:
            max_score_idx = np.argmax(scores)
            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            height, width, _ = image.shape
            box_coords = to_image_coords(boxes[max_score_idx], height, width)
            box_coords = np.floor(box_coords).astype(int)
            print(box_coords)
            
        except ValueError:
            return 4 # UNKNOWN, no traffic lights in sight
        
        # TODO: Get the bounding box with the highest score, resize to classification net input size, normalize, feed to classification net
        
        img = np.array(cv2.resize(image, (self.input_width, self.input_height)))
        # Normalize
        img = img/255.0
        
        imshape = img.shape

        img = np.reshape(img, (1, imshape[0], imshape[1], imshape[2]))
        #print(img.max())
        #print(img.min())
        #print(img.shape)
        
        #t1 = time.time()
        pred = self.sess.run(self.pred, feed_dict = {self.img: img, self.keep_prob: 1.0})[0]
        #logits = self.sess.run(self.logits, feed_dict = {self.img: img, self.keep_prob: 1.0})
        
        #print(logits)
        #print(time.time()-t1)
        
        if (pred == 3):
            pred = 4 # UNKNOWN
        
        return pred
        
        
        
        
        
        
