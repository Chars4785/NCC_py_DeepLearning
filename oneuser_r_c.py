import tensorflow as tf
import numpy as np
import math as m
import json
from collections import OrderedDict

file_data = OrderedDict()

tf.set_random_seed(777)
start = True

nb_classes = 5
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W1 = tf.get_variable("W1", shape=[4, 20], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([20]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[20, 20], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([20]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[20, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([5]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
"""
def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


def rev_normalize(data, alist):
    result = np.min(alist) + data * (np.max(alist) - np.min(alist))
    return result


def main():
    xy = np.loadtxt('user_dataset3.csv', delimiter=',', dtype=np.float32)
    t_y = xy[:, [0]]
    #xy = normalize(xy)
    x_data = xy[:, 1:5]
    y_data = xy[:, [0]]

    print(x_data)
    print(y_data)

    # train/test split
    test_size = int(len(y_data) * 0.5)
    train_size = len(y_data) - test_size
    trainX = np.array(x_data[0:train_size])
    testX = np.array(x_data[train_size:len(x_data)])
    trainY = np.array(y_data[0:train_size])
    testY = np.array(y_data[train_size:len(y_data)])

    print(trainX)
    print(trainY)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(50000):
            sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
            if step % 500 == 0:
                loss, acc= sess.run([cost, accuracy], feed_dict={X: trainX, Y: trainY})
                print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

        pred = sess.run(prediction, feed_dict={X: testX})
        correct_prediction = tf.equal(pred, testY)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        a = sess.run(acc, feed_dict={X: trainX, Y: trainY})
        #print("Test:", pred)
        #print("Real:", testY)
        print("Accuracy:", a)
        #pred = rev_normalize(pred, t_y)
        #print("Test:", pred)
        #print("type:", type(pred))

        fr = np.loadtxt('restaurant_dataset.csv', delimiter=',', dtype=np.str)
        pred_string = np.str(pred[0])
        print(fr)
        print(pred_string)
        f_rname = fr[fr[:, 1] == pred_string, [0]]
        f_data = fr[fr[:, 1] == pred_string, 2:4]
        fd = float(f_data[0, 0])
        f_x = []
        f_y = []
        minindex = 0
        f_x.insert(0, float(f_data[0, 0]))
        print(f_x)
        for i in range(len(f_data)):
            f_x.insert(i, float(f_data[i, 0]))
            f_y.insert(i, float(f_data[i, 1]))

        user_location = [100, 100]

        min_distance_x = pow((f_x[0] - user_location[0]), 2)
        min_distance_y = pow((f_y[0] - user_location[1]), 2)
        min_distance = min_distance_x + min_distance_y

        for i in range(len(f_data)):
            distance_x = pow((f_x[i] - user_location[0]), 2)
            distance_y = pow((f_y[i] - user_location[1]), 2)
            if (min_distance > distance_x + distance_y):
                min_distance_x = distance_x + distance_y
                minindex = i

        min_distance = m.sqrt(min_distance)

        print(f_rname[minindex] + "을 추천합니다")
        file_data["output"] = f_rname[minindex]
        with open('Output.json','w',encoding="utf=8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent="\t")
"""
        builder = tf.saved_model.builder.SavedModelBuilder("C:/Users/user/Desktop/py/serve")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()
"""

while start == True:
    main()
    start = False

