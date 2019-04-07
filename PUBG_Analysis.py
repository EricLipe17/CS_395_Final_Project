# This file will contain the neural network that analyzes the PUBG Dataset.

import os
import tensorflow as tf
import pandas as pd
import numpy as np

# Turning warning and info messages off:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Importing Data:
print("Importing Data:")
pd.set_option('display.max_columns', None)
train_data = pd.read_csv("D:\\PUBG_preprocessed_training_data.csv")
validation_data = pd.read_csv("D:\\PUBG_preprocessed_validation_data.csv")

print("\nMapping Validation Data:")
validation_targets = pd.DataFrame(validation_data['winPlacePerc'])
validation_inputs = validation_data.drop(["Unnamed: 0", "Id", "winPlacePerc"], axis=1)

print("\nMapping Train Data")
train_IDS = pd.DataFrame(train_data["Id"])
train_targets = pd.DataFrame(train_data["winPlacePerc"])
train_inputs = train_data.drop(["Unnamed: 0", "Id", "winPlacePerc"], axis=1)

print("\nBegin Training: ")


# Setting Constants:
input_size = 66
output_size = 1
hidden_layer_size = 150

# Building the Model:
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights_1 = tf.get_variable("Weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("Biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("Weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("Biases_2", [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("Weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("Biases_3", [hidden_layer_size])
outputs_3 = tf.nn.relu(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("Weights_4", [hidden_layer_size, hidden_layer_size])
biases_4 = tf.get_variable("Biases_4", [hidden_layer_size])
outputs_4 = tf.nn.relu(tf.matmul(outputs_3, weights_4) + biases_4)

weights_5 = tf.get_variable("Weights_5", [hidden_layer_size, hidden_layer_size])
biases_5 = tf.get_variable("Biases_5", [hidden_layer_size])
outputs_5 = tf.nn.relu(tf.matmul(outputs_4, weights_5) + biases_5)

weights_final = tf.get_variable("Weights_Final", [hidden_layer_size, output_size])
biases_final = tf.get_variable("Biases_Final", [output_size])
output = tf.nn.relu(tf.matmul(outputs_5, weights_final) + biases_final)

mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=output)
abs_mean_loss = tf.reduce_mean(tf.losses.absolute_difference(labels=targets, predictions=output))

optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(abs_mean_loss)

# Initializing the session for the Model to run:
sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

prev_validation_loss = 9999999

sess.run(initializer)
print("Running Algorithm: ")
epoch = 1
while True:
    _, square_loss, absolute_loss = sess.run([optimize, mean_loss, abs_mean_loss],
                                             feed_dict={inputs: train_inputs, targets: train_targets})

    validation_square_loss, validation_absolute_loss = sess.run([mean_loss, abs_mean_loss],
                                                                feed_dict={inputs: validation_inputs,
                                                                           targets: validation_targets})

    print('Epoch ' + str(epoch) +
          '. Training Square Loss: ' + '{0:.5f}'.format(square_loss) +
          '. Training Abs Loss: ' + '{0:.5f}'.format(absolute_loss) +
          '. Validation Square Loss: ' + '{0:.5f}'.format(validation_square_loss) +
          '. Validation Abs Loss: ' + '{0:.5f}'.format(validation_absolute_loss))

    if validation_absolute_loss > prev_validation_loss:
        print("Early Stopping Activated.\n")
        break

    prev_validation_loss = validation_absolute_loss
    epoch += 1

print("End of Training.")
sess.close()

