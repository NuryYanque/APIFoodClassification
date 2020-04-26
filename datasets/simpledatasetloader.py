# import the necessary packages
import numpy as np
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def preprocess(self, image):
		# initialize the list of features and labels
		data = []
		
		# image = cv2.imread(imagePath)

		# check to see if our preprocessors are not None
		if self.preprocessors is not None:
			# loop over the preprocessors and apply each to
			# the image
			for p in self.preprocessors:
				image = p.preprocess(image)

		# treat our processed image as a "feature vector"
		data.append(image)

		return np.array(data)
