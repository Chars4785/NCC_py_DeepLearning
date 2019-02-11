import pandas as pd
from sqlalchemy import create_engine
import tensorflow as tf
import numpy as np
import sys

age = int(sys.argv[1])
weather = int(sys.argv[2])
sex = int(sys.argv[3])

engine = create_engine('mysql+pymysql://server_address')

data = pd.read_sql_query('select * from user_group', engine)

user_Age = data['user_Age']
Weather = data['Weather']
user_Sex = data['user_Sex']
Category = data['Category']

tf.set_random_seed(777)

x_data = []
y_data = []

for i in range(len(user_Age)):
    x = []
    x.append(int(user_Age[i]))
    x.append(int(Weather[i]))
    x.append(int(user_Sex[i]))
    x_data.insert(i,x)

for i in range(len(Category)):
    y = []
    y.append(int(Category[i]))
    y_data.insert(i,y)

nb_classes = 4

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W1 = tf.get_variable("W1", shape=[3, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([20]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[20, 20],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([20]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[20, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([4]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                X: x_data, Y: y_data})
            #print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    pred = sess.run(prediction, feed_dict={X: [[age,weather,sex]]})
    print("Predicted Category: " + str(pred[0]))


