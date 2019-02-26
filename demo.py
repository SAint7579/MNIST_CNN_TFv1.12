import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
#Importing MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
#Creating MNIST object
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#Importing matplotlib and numpy
import matplotlib.pyplot as plt
import numpy as np

with tf.Session() as sess:  
    #Loading model and the graphs
    saver = tf.train.import_meta_graph('MNIST_CNN/model-1000.meta')
    saver.restore(sess,'MNIST_CNN/model-1000')
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    hold_prob = graph.get_tensor_by_name("hold_prob:0")   
    y_pred = graph.get_tensor_by_name("Prediction:0")
    softmax = tf.nn.softmax(y_pred,name="Softmax")
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for l in range(1, columns*rows +1):
        ind = int(np.random.randint(low= 0,high =500))
        fig.add_subplot(rows, columns, l)
        plt.imshow(mnist.test.images[ind-1:ind].reshape(28,28),cmap= 'gist_gray')
        pred = (sess.run(tf.argmax(softmax[0]),feed_dict = {x: mnist.test.images[ind-1:ind], y_true: mnist.test.labels[ind-1:ind], hold_prob:1.0}))
        print("Predicted Value: "+ str(pred) + "\tActual Value: "+str(list(mnist.test.labels[ind-1:ind][0]).index(1)))
    plt.show()
