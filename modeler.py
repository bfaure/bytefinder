#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import time, os

# Third-party Libraries
import numpy as np
np.random.seed(1)

from keras.models import Graph, model_from_json
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

###############################################################################

# Save model
def save(model, classes):

	print("Saving Model...")
	t = int(time.time())
	os.makedirs('models/%s' % t)
	open('models/%s/meta.json' % t, 'w').write(model.to_json())
	model.save_weights('models/%s/data.h5' % t, overwrite=True)
	open('models/%s/classes.txt' % t, 'w').write(', '.join(classes))

# Load model
def load(name):

	print("Loading Model...")
	model = model_from_json(open('models/%s/meta.json' % name).read())
	model.load_weights('models/%s/data.h5' % name)
	classes = open('models/%s/classes.txt' % name).read().split(', ')
	return model, classes

# Use model
def run(model, inMatrix):

	print("Running Model...")
	return model.predict({'input':inMatrix})['output']

# Build model
def build(inShape, targShape):

	print("Building Model...")
	model = Graph()
	model.add_input(name='input', input_shape=inShape[1:])
	model.add_node(LSTM(output_dim=targShape[-1],  return_sequences=True), name='forward', input='input')
	model.add_node(LSTM(output_dim=targShape[-1],  return_sequences=True, go_backwards=True), name='backward', input='input')
	model.add_output(name='output', inputs=['forward', 'backward'], merge_mode='ave')
	return model

# Train model
def train(model, inMatrix, targMatrix):

	print("Compiling Model...")
	model.compile(loss={'output':'mse'}, optimizer='rmsprop')
	print("Training Model...")
	model.fit({'input': inMatrix, 'output': targMatrix}, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1)
	return model