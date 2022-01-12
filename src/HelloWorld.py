import platform
import tensorflow as tf
print('VERSION',platform.python_version())
a = 2
b = 3
c = tf.add(a, b, name='Add')
print(c)
sess = tf.Session()
print(sess.run(c))
sess.close()