import sys
import nltk
import numpy as np

display_graphs = False # Boolean flag for displaying graphs
vocabulary = {} # A dictionary of all the unique words in the corpus

def load_parsed_data():
	"""
	Loads the train, test, and validation sets

	Returns:
		inputs_train   the input train set
		targets_train  the target train set
		inputs_valid   the input validation set
		targets_valid  the target validation set
		inputs_test    the input test set
		targets_test   the target test set
	"""
	print('loading parsed dataset')
	inputs_train  = np.load('../parsed_data/inputs_train.npy')
	targets_train = np.load('../parsed_data/targets_train.npy')
	inputs_valid  = np.load('../parsed_data/inputs_valid.npy')
	targets_valid = np.load('../parsed_data/targets_valid.npy')
	inputs_test   = np.load('../parsed_data/inputs_test.npy')
	targets_test  = np.load('../parsed_data/targets_test.npy')
	print('loaded parsed dataset')

	return inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test

def trained_model_exists():
	"""
	Checks to see if the extracted features for the Naive Bayes
	models are saved.

	Returns:
		boolean  True iff file 'features' exists
	"""
	return False # TODO: implement later when saving features dictionary

def load_features():
	"""
	Loads the extracted features for each data set

	Returns:
		train_features  a dictionary of the features in the train set
		valid_features  a dictionary of the features in the validation set
		test_features   a dictionary of the features in the test set
	"""
	pass # TODO: implement later

def build_vocabulary_list(inputs):
	"""
	Builds a list of unique words in the dataset

	Returns:
		vocabulary  a list of all the unique words in the corpus
	"""
	print('building vocabulary of words in the corpus')
	global vocabulary

	for tweet in inputs:
		for word in str(tweet).split():
			vocabulary[word] = word

	print('built vocabulary of words in the corpus')
	return vocabulary


def train_model(inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test):
	"""
	Trains the model. For Naive Bayes this is just extracting the features
	from the input data.

	Returns:
		train_features   a dictionary of word presence in the entire input
		                 dataset for each tweet
		                 {'contains(lol)': False, 'contains(jbiebs)': True, ...}

		valid_features   a dictionary of word presence in the entire input
		                 dataset for each tweet
		                 {'contains(lol)': False, 'contains(jbiebs)': True, ...}

		test_features    a dictionary of word presence in the entire input
		                 dataset for each tweet
		                 {'contains(lol)': False, 'contains(jbiebs)': True, ...}
	"""
	inputs = np.hstack((inputs_train, inputs_valid, inputs_test))
	vocabulary = build_vocabulary_list(inputs)

	print('training model')
	train_features = [(build_features(inputs_train[i]), targets_train[i]) for i in range(len(inputs_train))]
	valid_features = [(build_features(inputs_valid[i]), targets_valid[i]) for i in range(len(inputs_valid))]
	test_features  = [(build_features(inputs_test[i]), targets_test[i]) for i in range(len(inputs_test))]

	return train_features, valid_features, test_features

def main():
	"""
	CLI Arguments allowed:
		display_graphs  Displays graphs
		retrain         Trains a new model
	"""

	inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test = load_parsed_data()

	if 'display_graphs' in sys.argv:
		display_graphs = True

	if not trained_model_exists() or 'retrain' in sys.argv:
		train_features, valid_features, test_features = train_model(
			inputs_train,
			targets_train,
			inputs_valid,
			targets_valid,
			inputs_test,
			targets_test
		)
	else:
		train_features, valid_features, test_features = load_features()

if __name__ == "__main__": main()
