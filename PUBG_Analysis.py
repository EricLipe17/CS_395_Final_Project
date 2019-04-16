import tensorflow as tf
import pandas as pd
import numpy as np
from time import localtime, strftime
from PUBG_Batching_Class import PUBG_Data_Reader

batch_size = 50

train_data = PUBG_Data_Reader('PUBG_preprocessed_training_data.csv', batch_size)
validation_data = PUBG_Data_Reader('PUBG_preprocessed_validation_data.csv')


# Setting Constants
input_size = 66
output_size = 1
hidden_layer_size = 50
# Update, 5 layers still seems to be optimal with 50 nodes per layer. Now that we added batch training we got the loss
# to be 0.0662424117. Again, Relu is still the best.

# Building the Model
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])
# learning_rate = tf.placeholder(tf.float32, [])

weights_1 = tf.get_variable("Weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("Biases_1", [hidden_layer_size])
outputs_1 = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("Weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("Biases_2", [hidden_layer_size])
outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("Weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("Biases_3", [hidden_layer_size])
outputs_3 = tf.nn.sigmoid(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("Weights_4", [hidden_layer_size, hidden_layer_size])
biases_4 = tf.get_variable("Biases_4", [hidden_layer_size])
outputs_4 = tf.nn.sigmoid(tf.matmul(outputs_3, weights_4) + biases_4)

weights_5 = tf.get_variable("Weights_5", [hidden_layer_size, hidden_layer_size])
biases_5 = tf.get_variable("Biases_5", [hidden_layer_size])
outputs_5 = tf.nn.sigmoid(tf.matmul(outputs_4, weights_5) + biases_5)

# weights_6 = tf.get_variable("Weights_6", [hidden_layer_size, hidden_layer_size])
# biases_6 = tf.get_variable("Biases_6", [hidden_layer_size])
# outputs_6 = tf.nn.relu(tf.matmul(outputs_5, weights_6) + biases_6)
#
# weights_7 = tf.get_variable("Weights_7", [hidden_layer_size, hidden_layer_size])
# biases_7 = tf.get_variable("Biases_7", [hidden_layer_size])
# outputs_7 = tf.nn.relu(tf.matmul(outputs_6, weights_7) + biases_7)
#
# weights_8 = tf.get_variable("Weights_8", [hidden_layer_size, hidden_layer_size])
# biases_8 = tf.get_variable("Biases_8", [hidden_layer_size])
# outputs_8 = tf.nn.relu(tf.matmul(outputs_7, weights_8) + biases_8)
#
# weights_9 = tf.get_variable("Weights_9", [hidden_layer_size, hidden_layer_size])
# biases_9 = tf.get_variable("Biases_9", [hidden_layer_size])
# outputs_9 = tf.nn.relu(tf.matmul(outputs_8, weights_9) + biases_9)
#
# weights_10 = tf.get_variable("Weights_10", [hidden_layer_size, hidden_layer_size])
# biases_10 = tf.get_variable("Biases_10", [hidden_layer_size])
# outputs_10 = tf.nn.relu(tf.matmul(outputs_9, weights_10) + biases_10)

# weights_11 = tf.get_variable("Weights_11", [hidden_layer_size, hidden_layer_size])
# biases_11 = tf.get_variable("Biases_11", [hidden_layer_size])
# outputs_11 = tf.nn.relu(tf.matmul(outputs_10, weights_11) + biases_11)

# weights_12 = tf.get_variable("Weights_12", [hidden_layer_size, hidden_layer_size])
# biases_12 = tf.get_variable("Biases_12", [hidden_layer_size])
# outputs_12 = tf.nn.relu(tf.matmul(outputs_11, weights_12) + biases_12)

# weights_13 = tf.get_variable("Weights_13", [hidden_layer_size, hidden_layer_size])
# biases_13 = tf.get_variable("Biases_13", [hidden_layer_size])
# outputs_13 = tf.nn.relu(tf.matmul(outputs_12, weights_13) + biases_13)

# weights_14 = tf.get_variable("Weights_14", [hidden_layer_size, hidden_layer_size])
# biases_14 = tf.get_variable("Biases_14", [hidden_layer_size])
# outputs_14 = tf.nn.relu(tf.matmul(outputs_13, weights_14) + biases_14)

# weights_15 = tf.get_variable("Weights_15", [hidden_layer_size, hidden_layer_size])
# biases_15 = tf.get_variable("Biases_15", [hidden_layer_size])
# outputs_15 = tf.nn.relu(tf.matmul(outputs_14, weights_15) + biases_15)

# weights_16 = tf.get_variable("Weights_16", [hidden_layer_size, hidden_layer_size])
# biases_16 = tf.get_variable("Biases_16", [hidden_layer_size])
# outputs_16 = tf.nn.relu(tf.matmul(outputs_15, weights_16) + biases_16)

# weights_17 = tf.get_variable("Weights_17", [hidden_layer_size, hidden_layer_size])
# biases_17 = tf.get_variable("Biases_17", [hidden_layer_size])
# outputs_17 = tf.nn.relu(tf.matmul(outputs_16, weights_17) + biases_17)

# weights_18 = tf.get_variable("Weights_18", [hidden_layer_size, hidden_layer_size])
# biases_18 = tf.get_variable("Biases_18", [hidden_layer_size])
# outputs_18 = tf.nn.relu(tf.matmul(outputs_17, weights_18) + biases_18)

# weights_19 = tf.get_variable("Weights_19", [hidden_layer_size, hidden_layer_size])
# biases_19 = tf.get_variable("Biases_19", [hidden_layer_size])
# outputs_19 = tf.nn.relu(tf.matmul(outputs_18, weights_19) + biases_19)

# weights_20 = tf.get_variable("Weights_20", [hidden_layer_size, hidden_layer_size])
# biases_20 = tf.get_variable("Biases_20", [hidden_layer_size])
# outputs_20 = tf.nn.relu(tf.matmul(outputs_19, weights_20) + biases_20)

weights_final = tf.get_variable("Weights_Final", [hidden_layer_size, output_size])
biases_final = tf.get_variable("Biases_Final", [output_size])
output = tf.nn.sigmoid(tf.matmul(outputs_5, weights_final) + biases_final)

mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=output)
abs_mean_loss = tf.reduce_mean(tf.losses.absolute_difference(labels=targets, predictions=output))

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)


# Keeping track of the time pre-training:
print("\n\nStart Time: " + str(strftime("%Y-%m-%d %H:%M:%S", localtime())))

# Initializing the session for the Model to run:
sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

prev_validation_loss = 9999999

# lr_schedule = [0.001, 0.00001, 0.00000001]

sess.run(initializer)
print("\n\nRunning Algorithm: ")
epoch = 1
while True:
#     _, absolute_loss, squared_loss = sess.run([optimize, abs_mean_loss, mean_loss], feed_dict={inputs: train_inputs, targets: train_targets})

#     validation_absolute_loss = sess.run(abs_mean_loss, feed_dict={inputs: validation_inputs, targets: validation_targets})

    curr_square_epoch_loss = 0.
    curr_abs_epoch_loss = 0.

    for input_batch, target_batch in train_data:
        _, batch_loss, batch_abs_mean_loss = sess.run([optimize, mean_loss, abs_mean_loss],
                                                      feed_dict={inputs: input_batch, targets: target_batch})

        curr_square_epoch_loss += batch_loss
        curr_abs_epoch_loss += batch_abs_mean_loss

    curr_square_epoch_loss /= train_data.batch_count
    curr_abs_epoch_loss /= train_data.batch_count

    validation_square_loss = 0.
    validation_absolute_loss = 0.

    for input_batch, target_batch in validation_data:
        validation_square_loss, validation_absolute_loss = sess.run([mean_loss, abs_mean_loss],
                                                                    feed_dict={inputs: input_batch,
                                                                               targets: target_batch})

    print('Epoch ' +str(epoch) +
         '. Training Loss: ' + '{0:.10f}'.format(curr_square_epoch_loss) +
         '. Training Abs Loss: ' + '{0:.10f}' .format(curr_abs_epoch_loss) +
         '. Validation Loss: ' + '{0:.10f}'.format(validation_square_loss) +
         '. Validation Abs Loss: ' + '{0:.10f}'.format(validation_absolute_loss))

    if validation_absolute_loss > prev_validation_loss:
        print("Early Stopping Activated.\n")
        break

    prev_validation_loss = validation_absolute_loss
    epoch += 1

print("End of Training.")
print("End Time: " + str(strftime("%Y-%m-%d %H:%M:%S", localtime())))
sess.close()



