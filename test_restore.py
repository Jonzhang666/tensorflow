import tensorflow as tf
import numpy as np
#tf.reset_default_graph()
# 先建立 W, b 的容器
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    #tf.reset_default_graph()
    # 提取变量
    saver.restore(sess, "test_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
