import tensorflow as tf
import numpy as np
import SimpleITK

from tensorflow.examples.tutorials.mnist import input_data
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
print('Getting data.........................')
mnist = input_data.read_data_sets('/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/MNIST_data')
#mnist = input_data.read_data_sets('MNIST_data')
print('Getting data.........................DONE')

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)


NUM_GPUS = 2
#
#strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
#config = tf.estimator.RunConfig(train_distribute=strategy)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
classifier = tf.compat.v1.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.compat.v1.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    #config=config,
    #model_dir="./simple_model"
    model_dir="/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/simple_model_gx"
, loss_reduction=tf.keras.losses.Reduction.SUM)

# Define the training inputs
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)


classifier.train(input_fn=train_input_fn, steps=100000)

# Define the test inputs
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

