#####################################
# -*- coding: utf-8 -*-
# Javier Vega Garcia
# Convolucional Neural Network - CNN
#####################################
# ACTIVAR EL SERVICIO
#####################################
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json

def Modelo_RNA():
	print("Cargando modelo desde el disco ...")
	file_model = "../model/cnn_jvg.json"
	file_pesos = "../model/cnn_jvg.h5"

	json_file = open(file_model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(file_pesos)
	loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	print("Modelo cargado !!!")
	graph = tf.get_default_graph()
	return loaded_model, graph
