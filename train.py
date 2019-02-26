import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#Importing MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
#Creating MNIST object
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
from helper_func import *
tf.reset_default_graph()
#Initializing placeholders
x = tf.placeholder(tf.float32,shape = [None,784],name='x')       #For the images
y_true = tf.placeholder(tf.float32,shape = [None,10],name='y_true')   #For true data
hold_prob = tf.placeholder(tf.float32,name='hold_prob')                  #Hold probability for dropout

#Creating CNN layers
x_image = tf.reshape(x,[-1,28,28,1])
convo_1 = conv_layer(x_image,shape=[5,5,1,32])          #32 features of 1 layer for each 5,5 patch
pool_1 = max_pooling_2by2(convo_1)
convo_2 = conv_layer(pool_1, shape = [5,5,32,64])       #64 features of previous 32 layers for each 5,5 patch
pool_2 = max_pooling_2by2(convo_2)

#Flattening the CNN output
flattened = tf.reshape(pool_2, [-1,7*7*64])             #2 max-poolings on a 28x28 image = 7x7 image

#Creating DNN layers
dnn1 = tf.nn.relu(dnn_layer(flattened,1024))                #1024 neurons in the DNN layer
dropout1 = tf.nn.dropout(dnn1, keep_prob = hold_prob) #Adding dropout
dnn2 = tf.nn.relu(dnn_layer(dropout1,1024))                 #1024 neurons in the DNN layer
dropout2 = tf.nn.dropout(dnn2, keep_prob = hold_prob)       #Adding dropout

#Making output
y_pred = dnn_layer(dropout2,10,name='Prediction')

#Loss function for optimization
cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

#Initializaing optimizer and trainer
optim = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optim.minimize(cross_ent)

init = tf.global_variables_initializer()    #Global variable initializer

sav = tf.train.Saver()  #For saving the model

#Training the model
with tf.Session() as sess:
    steps = 1000    #Number of steps
    sess.run(init)  #Initializing variables
    for i in range(steps):
        #Fetching batches for training
        batch_x , batch_y = mnist.train.next_batch(100)
        sess.run(train,feed_dict = {x:batch_x, y_true: batch_y, hold_prob: 0.5})
        #Displaying accuracy after 100 steps
        if i%100 == 0:   
            print("ON STEP:",i,end="\t")
            print("ACCURACY: ",end='')
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1)) #Tensor for correct predictions
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))           #Accuracy ratio
            print(sess.run(acc,feed_dict = {x: mnist.test.images[:1000], y_true: mnist.test.labels[:1000], hold_prob:1.0}))
    #Saving the model
    sav.save(sess,"MNIST_CNN/model",global_step=1000)
