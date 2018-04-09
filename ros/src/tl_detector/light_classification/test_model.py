#!/usr/bin/env python
import tensorflow as tf
import cv2
import numpy as np
import os
import time
import glob
import pandas as pd

class TLClassifierTester(object):
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
        #logits = self.sess.run(self.logits, feed_dict = {self.img: img, self.keep_prob: 1.0})
        #logits = sess.run(self.logits, feed_dict = {self.img: img, self.keep_prob: 1.0})
        #print(logits)
        #print(time.time()-t1)
        
        #return 4
        #if (pred == 3):
        #    pred = 4 # UNKNOWN
        
        return pred
        
if __name__ == '__main__':
    # Load an arbitrary number of test images
    num_test_images = 1000
    training_file = '../../../../trainingimages/'
    
    red_img_paths = glob.glob(training_file + '*RED.png')
    green_img_paths = glob.glob(training_file + '*GREEN.png')
    yellow_img_paths = glob.glob(training_file + '*YELLOW.png')
    unknown_img_paths = glob.glob(training_file + '*UNKNOWN.png')

    img_paths = np.array(red_img_paths + green_img_paths + yellow_img_paths + unknown_img_paths)
    labels = np.array([0] * len(red_img_paths) + [1] * len(yellow_img_paths) + [2] * len(green_img_paths) + [3] * len(unknown_img_paths))
    
    indices = np.random.choice(np.arange(len(labels)), num_test_images, replace=False)
    img_paths_sampled = img_paths[indices]
    labels_sampled = labels[indices]
    
    tlctester = TLClassifierTester()
    predictions = []
    
    for i, (img_path, label) in enumerate(zip(img_paths_sampled, labels_sampled)):
        #print img_path, label
        #print("progress: {}/{}".format(i, num_test_images))
        img = cv2.imread(img_path)
        prediction = tlctester.get_classification(img)
        predictions.append(prediction)
        if i%10 == 0:
            print("progress: {}/{}".format(i, num_test_images))
        
    predictions = np.array(predictions)
    #print(predictions.dtype)
    #print(labels_sampled.dtype)
    #print(predictions == labels_sampled)
    accuracy = np.sum(predictions == labels_sampled)/float(num_test_images)
    print("Accuracy: {}".format(accuracy))
    
    y_actu = pd.Series(labels_sampled, name='Actual')
    y_pred = pd.Series(predictions, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
        
    




















