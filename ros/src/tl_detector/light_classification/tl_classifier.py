from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.sess = tf.Session()
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        img_key = 'img'
        kp_key = 'kp'
        pred_key = 'pred'

        export_path = '/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/model'

        meta_graph_def = tf.saved_model.loader.load(
                            self.sess,
                            [tf.saved_model.tag_constants.SERVING],
                            export_path)
        signature = meta_graph_def.signature_def

        img_tensor_name = signature[signature_key].inputs[img_key].name
        kp_tensor_name = signature[signature_key].inputs[kp_key].name
        pred_tensor_name = signature[signature_key].outputs[pred_key].name

        self.img = self.sess.graph.get_tensor_by_name(img_tensor_name)
        self.keep_prob = self.sess.graph.get_tensor_by_name(kp_tensor_name)
        self.pred = self.sess.graph.get_tensor_by_name(pred_tensor_name)
        
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        img = np.array(cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25))
        imshape = img.shape

        img = np.reshape(img, (1, imshape[0], imshape[1], imshape[2])) 

        pred = self.sess.run(self.pred, feed_dict = {self.img: img, self.keep_prob: 1.0})[0]

        if (pred == 3):
            pred = 4 # UNKNOWN
        
        return pred
