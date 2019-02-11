import tensorflow as tf
import numpy as np

start = input("현재 위치:")

test_sample =[]

#충북여행코스-충북나드리
sample1 = [['수옥폭포', '한지체험박물관', '화양구곡', '선유구곡', '쌍곡구곡'],
           ['상수허브랜드','문의문화재단지','청남대'],
           ['도담삼봉','다누리아쿠아리움','구경시장','만천하스카이워크'],
           ['법주사','삼년 산성','수암골'],
           ['의림지','청풍문화재단지','청풍랜드','산야초마을','정방사']]

#입력받은 현재위치와 같은 시작위치를 가진 여행코스 검색
rt = int
for k in range(0,len(sample1),1):
    for j in range(0, len(sample1[k]), 1):
        if(sample1[k][j] == start):
            test_sample = sample1[k]
            rt = j
            break;
        
a_len = len(test_sample)

x_data = []
x_one_hot = []
y_data = []
x_one_hot1 = []
x_data1 = []
y_data1 = []

#x data를 one_hot encoding하는 과정
for x in range(0, (a_len)-1, 1):
    x_data1.insert(x,x)

x_one_hot1 = [[0 for i in  range(a_len)]for j in range((a_len)-1)]

for i in range((a_len)-1):
    x_one_hot1[i][i] = 1

x_data.insert(0,x_data1)
x_one_hot.insert(0,x_one_hot1)

for y in range(1, a_len, 1):
    y_data1.insert(y,y)

y_data.insert(0,y_data1)

input_dim = a_len
hidden_size = a_len
num_classes = a_len
sequence_length = (a_len)-1 
learning_rate = 0.1
batch_size = 1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  
Y = tf.placeholder(tf.int32, [None, sequence_length]) 

cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})

        result_str = [test_sample[c] for c in np.squeeze(result)]


    print("다음 여행지로 ",result_str[rt],"을 추천합니다")

#없는 장소명입력
#마지막 장소명입력
#중복된 장소명입력
