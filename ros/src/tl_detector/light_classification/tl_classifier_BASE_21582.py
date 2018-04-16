#!/usr/bin/env python
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import os
import time

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
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

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
        logits = self.sess.run(self.logits, feed_dict = {self.img: img, self.keep_prob: 1.0})
        
        print(logits)
        #print(time.time()-t1)
        
        if (pred == 3):
            pred = 4 # UNKNOWN
        
        return pred
