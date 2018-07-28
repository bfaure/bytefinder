#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import time, os

# Third-party Libraries
import numpy as np
np.random.seed(1)

from keras.models import Model, model_from_json
from keras.layers import Input, TimeDistributed, Dense
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
	classes = open('models/%s/classes.txt' % name).read().split(', ')
	model = model_from_json(open('models/%s/meta.json' % name).read())
	model.load_weights('models/%s/data.h5' % name)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	return model, classes

# Use model
def run(model, inMatrix):

	window = 400
	classes = 2

	print("Running Model...")
	indices = range(0, inMatrix.shape[1], window//2)
	slices = [[x[y:y+window] for y in indices] for x in inMatrix]
	outMatrix = np.empty((inMatrix.shape[0], inMatrix.shape[1], classes), dtype=np.dtype('float32'))
	for sample in xrange(len(slices)):
		for piece in xrange(len(slices[sample])):
			if not piece%100:
				print("%0.2f%% Done" % (100.0*piece/len(slices[sample])))
			slices[sample][piece] = np.pad(slices[sample][piece], ((0,window-slices[sample][piece].shape[0]),(0,0)), mode='constant')
			prediction = model.predict(np.expand_dims(slices[sample][piece],axis=0))[0]
			if piece > 0 and piece < len(slices[sample])-1:
				prediction[:window//2] = np.mean(np.array([prediction[:window//2],outMatrix[sample,indices[piece]:indices[piece]+window//2]]),axis=0)
				if piece==len(slices[sample])-1:
					prediction = prediction[:outMatrix.shape[1]-indices[piece]]
			if outMatrix.shape[1]-indices[piece] < window:
				prediction = prediction[:outMatrix.shape[1]-indices[piece]]
			outMatrix[sample,indices[piece]:indices[piece]+window] = prediction
	return outMatrix

# Build model
def build(inShape, targShape):

	print("Building Model...")
	input_ = Input(inShape[1:])
	forward = LSTM(output_dim=targShape[-1]*4, return_sequences=True)(input)
	backward = LSTM(output_dim=targShape[-1]*4, return_sequences=True, go_backwards=True)(input_)
	output = TimeDistributed(Dense(targShape[-1], activation='softmax'))(forward, backward)
	return Model(input=input_, output=output)

# Train model
def train(model, inMatrix, targMatrix):

	print("Compiling Model...")
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print("Training Model...")
	model.fit(inMatrix, targMatrix, nb_epoch=100, validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
	return model
