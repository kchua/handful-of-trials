# import tensorflow as tf

# print(tf.test.is_gpu_available())
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
# sess = tf.Session(config=config)

import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.Session(config=config).close()

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()
 
def calc():
    N = 15 # works for N <= 14
    a = 64
    b = 16
    X = np.random.rand(N, 11520, b, 1).astype(np.float32)
    # X = np.random.rand(10, 5, 24, 200 ).astype(np.float32)
    print(X.nbytes*1e-6, "MB")
    W = np.random.rand(N, 11520, a, b).astype(np.float32)
    # W = np.random.rand(10, 5, 32, 24).astype(np.float32)
    print(W.nbytes*1e-6, "MB")
    X_ = tf.constant(X, name="X-constant", dtype=tf.float32)
    W_ = tf.constant(W, name="W-constant", dtype=tf.float32)
 
    return tf.matmul(W_, X_, name="mymatmul")
    # return W_ @ X_
 
tf.reset_default_graph()
a = calc()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
b = sess.run(a)
sess.close()
print(b.shape)
