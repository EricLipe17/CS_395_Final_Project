import tensorflow as tf
from time import localtime, strftime
from PUBG_Batching_Class import PUBG_Data_Reader

batch_size = 100

train_data = PUBG_Data_Reader('PUBG_preprocessed_training_data.csv', batch_size)
validation_data = PUBG_Data_Reader('PUBG_preprocessed_validation_data.csv')

print(train_data.inputs_head())

# # Setting Constants
# input_size = 66
# output_size = 1
# hidden_layer_size = 400
#
# # Update 4/19/19: best predictor: 5 hidden layers with nodes staring at 200, 200, 100, 50, 25, 1.
# # Sigmoid activations, eta=0.001, batch size 100 and optimizing based off of abs_mean_loss.
#
# # Update: 4/20/19: Current best predictor is 5 hidden layers with nodes starting at 400, 400, 200, 100, 50, 1.
# # # Sigmoid activations, eta=0.001, batch size 100 and optimizing based off of abs_mean_loss. Loss got to
# # 0.06493.
#
# # Building the Model
# tf.reset_default_graph()
#
# inputs = tf.placeholder(tf.float32, [None, input_size])
# targets = tf.placeholder(tf.float32, [None, output_size])
#
# weights_1 = tf.get_variable("Weights_1", [input_size, hidden_layer_size])
# biases_1 = tf.get_variable("Biases_1", [hidden_layer_size])
# outputs_1 = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)
#
# weights_2 = tf.get_variable("Weights_2", [hidden_layer_size, hidden_layer_size])
# biases_2 = tf.get_variable("Biases_2", [hidden_layer_size])
# outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)
#
# weights_3 = tf.get_variable("Weights_3", [hidden_layer_size, hidden_layer_size // 2])
# biases_3 = tf.get_variable("Biases_3", [hidden_layer_size // 2])
# outputs_3 = tf.nn.sigmoid(tf.matmul(outputs_2, weights_3) + biases_3)
#
# weights_4 = tf.get_variable("Weights_4", [hidden_layer_size // 2, hidden_layer_size // 4])
# biases_4 = tf.get_variable("Biases_4", [hidden_layer_size // 4])
# outputs_4 = tf.nn.sigmoid(tf.matmul(outputs_3, weights_4) + biases_4)
#
# weights_5 = tf.get_variable("Weights_5", [hidden_layer_size // 4, hidden_layer_size // 8])
# biases_5 = tf.get_variable("Biases_5", [hidden_layer_size // 8])
# outputs_5 = tf.nn.sigmoid(tf.matmul(outputs_4, weights_5) + biases_5)
# #
# # weights_6 = tf.get_variable("Weights_6", [hidden_layer_size // 4, hidden_layer_size // 4])
# # biases_6 = tf.get_variable("Biases_6", [hidden_layer_size // 4])
# # outputs_6 = tf.nn.sigmoid(tf.matmul(outputs_5, weights_6) + biases_6)
# #
# # weights_7 = tf.get_variable("Weights_7", [hidden_layer_size // 4, hidden_layer_size // 8])
# # biases_7 = tf.get_variable("Biases_7", [hidden_layer_size // 8])
# # outputs_7 = tf.nn.sigmoid(tf.matmul(outputs_6, weights_7) + biases_7)
# #
# # weights_8 = tf.get_variable("Weights_8", [hidden_layer_size // 8, hidden_layer_size // 8])
# # biases_8 = tf.get_variable("Biases_8", [hidden_layer_size // 8])
# # outputs_8 = tf.nn.sigmoid(tf.matmul(outputs_7, weights_8) + biases_8)
# #
# # weights_9 = tf.get_variable("Weights_9", [hidden_layer_size // 8, hidden_layer_size // 16])
# # biases_9 = tf.get_variable("Biases_9", [hidden_layer_size // 16])
# # outputs_9 = tf.nn.sigmoid(tf.matmul(outputs_8, weights_9) + biases_9)
# #
# # weights_10 = tf.get_variable("Weights_10", [hidden_layer_size // 16, hidden_layer_size // 16])
# # biases_10 = tf.get_variable("Biases_10", [hidden_layer_size // 16])
# # outputs_10 = tf.nn.sigmoid(tf.matmul(outputs_9, weights_10) + biases_10)
#
# # weights_11 = tf.get_variable("Weights_11", [hidden_layer_size, hidden_layer_size])
# # biases_11 = tf.get_variable("Biases_11", [hidden_layer_size])
# # outputs_11 = tf.nn.sigmoid(tf.matmul(outputs_10, weights_11) + biases_11)
# #
# # weights_12 = tf.get_variable("Weights_12", [hidden_layer_size, hidden_layer_size])
# # biases_12 = tf.get_variable("Biases_12", [hidden_layer_size])
# # outputs_12 = tf.nn.sigmoid(tf.matmul(outputs_11, weights_12) + biases_12)
# #
# # weights_13 = tf.get_variable("Weights_13", [hidden_layer_size, hidden_layer_size])
# # biases_13 = tf.get_variable("Biases_13", [hidden_layer_size])
# # outputs_13 = tf.nn.sigmoid(tf.matmul(outputs_12, weights_13) + biases_13)
# #
# # weights_14 = tf.get_variable("Weights_14", [hidden_layer_size, hidden_layer_size])
# # biases_14 = tf.get_variable("Biases_14", [hidden_layer_size])
# # outputs_14 = tf.nn.sigmoid(tf.matmul(outputs_13, weights_14) + biases_14)
# #
# # weights_15 = tf.get_variable("Weights_15", [hidden_layer_size, hidden_layer_size])
# # biases_15 = tf.get_variable("Biases_15", [hidden_layer_size])
# # outputs_15 = tf.nn.sigmoid(tf.matmul(outputs_14, weights_15) + biases_15)
#
# # weights_16 = tf.get_variable("Weights_16", [hidden_layer_size, hidden_layer_size])
# # biases_16 = tf.get_variable("Biases_16", [hidden_layer_size])
# # outputs_16 = tf.nn.relu(tf.matmul(outputs_15, weights_16) + biases_16)
#
# # weights_17 = tf.get_variable("Weights_17", [hidden_layer_size, hidden_layer_size])
# # biases_17 = tf.get_variable("Biases_17", [hidden_layer_size])
# # outputs_17 = tf.nn.relu(tf.matmul(outputs_16, weights_17) + biases_17)
#
# # weights_18 = tf.get_variable("Weights_18", [hidden_layer_size, hidden_layer_size])
# # biases_18 = tf.get_variable("Biases_18", [hidden_layer_size])
# # outputs_18 = tf.nn.relu(tf.matmul(outputs_17, weights_18) + biases_18)
#
# # weights_19 = tf.get_variable("Weights_19", [hidden_layer_size, hidden_layer_size])
# # biases_19 = tf.get_variable("Biases_19", [hidden_layer_size])
# # outputs_19 = tf.nn.relu(tf.matmul(outputs_18, weights_19) + biases_19)
#
# # weights_20 = tf.get_variable("Weights_20", [hidden_layer_size, hidden_layer_size])
# # biases_20 = tf.get_variable("Biases_20", [hidden_layer_size])
# # outputs_20 = tf.nn.relu(tf.matmul(outputs_19, weights_20) + biases_20)
#
# weights_final = tf.get_variable("Weights_Final", [hidden_layer_size // 8, output_size])
# biases_final = tf.get_variable("Biases_Final", [output_size])
# output = tf.nn.sigmoid(tf.matmul(outputs_5, weights_final) + biases_final)
#
# mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=output)
# abs_mean_loss = tf.reduce_mean(tf.losses.absolute_difference(labels=targets, predictions=output))
#
# # Implementing an exponentially decaying learning rate.
# initial_learning_rate = 0.001
# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=0.96, decay_steps=10000, staircase=True)
#
# optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(abs_mean_loss, global_step=global_step)
#
#
# # Keeping track of the time pre-training:
# print("\n\nStart Time: " + str(strftime("%Y-%m-%d %H:%M:%S", localtime())))
#
# # Initializing the session for the Model to run:
# sess = tf.InteractiveSession()
#
# initializer = tf.global_variables_initializer()
#
# prev_validation_loss = 9999999
#
# # Implementing a saver
# saver = tf.train.Saver()
# save_path = "C:\\Users\\retic\Desktop\CS_395_Final_Project\\Curr_Best_Model.ckpt"
#
#
# sess.run(initializer)
# print("\n\nRunning Algorithm: ")
# epoch = 1
# counter = 0
# while True:
# #     _, absolute_loss, squared_loss = sess.run([optimize, abs_mean_loss, mean_loss], feed_dict={inputs: train_inputs, targets: train_targets})
#
# #     validation_absolute_loss = sess.run(abs_mean_loss, feed_dict={inputs: validation_inputs, targets: validation_targets})
#
#     curr_square_epoch_loss = 0.
#     curr_abs_epoch_loss = 0.
#
#     for input_batch, target_batch in train_data:
#         _, batch_loss, batch_abs_mean_loss = sess.run([optimize, mean_loss, abs_mean_loss],
#                                                       feed_dict={inputs: input_batch, targets: target_batch})
#
#         curr_square_epoch_loss += batch_loss
#         curr_abs_epoch_loss += batch_abs_mean_loss
#
#     curr_square_epoch_loss /= train_data.batch_count
#     curr_abs_epoch_loss /= train_data.batch_count
#
#     validation_square_loss = 0.
#     validation_absolute_loss = 0.
#
#     for input_batch, target_batch in validation_data:
#         validation_square_loss, validation_absolute_loss = sess.run([mean_loss, abs_mean_loss],
#                                                                     feed_dict={inputs: input_batch,
#                                                                                targets: target_batch})
#
#     print('Epoch ' +str(epoch) +
#          '. Training Loss: ' + '{0:.5f}'.format(curr_square_epoch_loss) +
#          '. Training Abs Loss: ' + '{0:.5f}' .format(curr_abs_epoch_loss) +
#          '. Validation Loss: ' + '{0:.5f}'.format(validation_square_loss) +
#          '. Validation Abs Loss: ' + '{0:.5f}'.format(validation_absolute_loss))
#
#     if validation_absolute_loss > prev_validation_loss:
#         counter += 1
#         print("Current Validation greater than Previous Validation: Counter incremented.")
#     else:
#         counter = 0
#         print("Counter Reset.")
#         saver.save(sess=sess, save_path=save_path)
#     if abs(validation_absolute_loss - prev_validation_loss) < 10**(-11):
#         print("Validation loss is hardly changing. \nEarly Stopping Activated.")
#         break
#     if counter > 2:
#         print("Early Stopping Activated.\n")
#         break
#
#     prev_validation_loss = validation_absolute_loss
#     epoch += 1
#
# print("End of Training.")
# print("Saving model to directory:")
# saver.save(sess=sess, save_path=save_path)
# print("End Time: " + str(strftime("%Y-%m-%d %H:%M:%S", localtime())))
# sess.close()



