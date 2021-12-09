import cv2
from cv2 import dnn
import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# load the model from SavedModel format
model = load_model('C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\cnn\\saved_models\\v1.0')

# get model TF graph
tf_model_graph = tf.function(lambda x: model(x))

# get concrete function
tf_model_graph = tf_model_graph.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# obtain frozen concrete function
frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
# get frozen graph
frozen_tf_func.graph.as_graph_def()

# save full tf model
tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                  logdir='C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\frozen_models',
                  name='v1.0_frozen_graph.pb',
                  as_text=False)

# cv2_net = cv2.dnn.readNetFromTensorflow("C:/Users/Robotics3/PycharmProjects/pythonProject/frozen_models/UGCNN.pb")
# print(cv2_net.getLayerNames())
