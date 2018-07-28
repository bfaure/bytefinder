#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

# Standard Libraries
import os

# Local Libraries
import modeler
import data

###############################################################################

def main():

	if not os.path.exists('data'):
		os.makedirs('data')

	if next(os.walk('data'))[1]:
		retrain = False
		if retrain:
			input_, target, classes = data.sample('data')
			model = modeler.build(input_.shape, target.shape)
			modeler.train(model, input_, target)
			modeler.save(model, classes)
		else:
			model, classes = modeler.load(sorted(os.listdir('models'))[-1])
			with open("data/harry-potter/Sorcerer's Stone.txt") as f:
				input_ = data.str2mat(f.read())
			output = modeler.run(model, input_)
			data.backtest(classes, input_, output)
	else:
		print("""\nNo data found.\nPut subfolders of files by class, within the 'data' folder.""")

if __name__ == "__main__":
	main()
