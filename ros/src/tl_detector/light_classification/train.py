import rospy
import tensorflow as tf
import numpy as np
import glob
import cv2
from tensorflow.contrib.layers import flatten
import time
import math

training_file = '../../../../trainingimages/'

red_img_paths = glob.glob(training_file + '*RED.png')
green_img_paths = glob.glob(training_file + '*GREEN.png')
yellow_img_paths = glob.glob(training_file + '*YELLOW.png')
unknown_img_paths = glob.glob(training_file + '*UNKNOWN.png')

img_paths = red_img_paths + green_img_paths + yellow_img_paths + unknown_img_paths
labels = [0] * len(red_img_paths) + [1] * len(yellow_img_paths) + [2] * len(green_img_paths) + [3] * len(unknown_img_paths)

print("UNKNWON label is 3!")

imgs = np.array(map(lambda x: cv2.resize(cv2.imread(x), (0,0), fx = 0.25, fy = 0.25), img_paths)) #this might take a while, maybe make this faster?
labels = np.array(labels)

imshape = imgs[0].shape
print(imshape)

print(len(img_paths))
print(len(imgs))
print(len(labels))
print(img_paths[0])
#print(imgs[0])
print(labels[0])

from sklearn.model_selection import train_test_split
train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels, test_size = 0.2)

EPOCHS = 3
BATCH_SIZE = 50

def tl_clf(x):

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    conv1_size = 11
    strides = [1, 1, 1, 1]

    # Layer 1: Convolutional. Input = 600x800x3. Output = conv1_out_height x conv1_out_width x 6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(conv1_size, conv1_size, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=strides, padding='VALID') + conv1_b

    conv1_out_height = math.ceil(float(imshape[0] - conv1_size + 1) / float(strides[1]))
    conv1_out_width = math.ceil(float(imshape[1] - conv1_size + 1) / float(strides[2]))

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Halves height and width
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    pool1_out_height = int(conv1_out_height // 2)
    pool1_out_width = int(conv1_out_width // 2)

    # Flatten. 
    fc0   = flatten(conv1)

    flatten_size = pool1_out_height * pool1_out_width * 6

    # Layer 3: Fully Connected. Input = flatten_size. Output = 100.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(flatten_size, 100), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(100))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1= tf.nn.relu(fc1)

    # Dropout.
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 5: Fully Connected. Input = 100. Output = 4.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(100, 4), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(4))
    logits = tf.matmul(fc1, fc2_W) + fc2_b
    
    return logits


x = tf.placeholder(tf.float32, (None, imshape[0], imshape[1], 3), name="img")
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
one_hot_y = tf.one_hot(y, 4)

rate = 0.001

logits = tl_clf(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

prediction = tf.argmax(logits, 1, name="pred")
correct_prediction = tf.equal(prediction, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


from sklearn.utils import shuffle

export_path = './model'
builder = tf.saved_model.builder.SavedModelBuilder(export_path)


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(train_imgs)
        
        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
        tensor_info_kp = tf.saved_model.utils.build_tensor_info(keep_prob)
        tensor_info_pred = tf.saved_model.utils.build_tensor_info(prediction)

        prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'img': tensor_info_x, 'kp': tensor_info_kp},
              outputs={'pred': tensor_info_pred},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            prediction_signature 
          },
        )

        print("Training...")
        print()
        for i in range(EPOCHS):
            print("EPOCH {} ...".format(i+1))
            t1 = time.time()
            train_imgs, train_labels = shuffle(train_imgs, train_labels)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = train_imgs[offset:end], train_labels[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})
    
            training_accuracy = evaluate(train_imgs, train_labels)
            t2 = time.time()
            dt = t2 - t1
            print("...took {} minutes, {} seconds".format(dt // 60, round(dt % 60)))
            print("Training Accuracy = {:.3f}".format(training_accuracy))
            print()
    
        test_accuracy = evaluate(test_imgs, test_labels)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

        builder.save()
        print("Model saved")





