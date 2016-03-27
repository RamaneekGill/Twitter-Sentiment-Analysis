import sys
import pickle
import os.path
import operator
import numpy as np
from sklearn.naive_bayes import MultinomialNB

### Global variables
display_graphs = False # Boolean flag for displaying graphs
vocabulary = {} # A dictionary of all the unique words in the corpus

### Change me to higher values for better accuracy!
NUM_FEATURES = 2000 # The number of most common words in the corpus to use as features
PERCENTAGE_DATA_SET_TO_USE = 0.1 # The percentage of the dataset to use


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
		boolean  True iff file 'data/model.pkl' exists
	"""
	return os.path.exists('data/model.pkl')

def load_trained_model():
	"""Loads and returns the trained model"""
	print('loading trained model')
	with open('data/model.pkl', 'rb') as input:
		classifier = pickle.load(input)
		print('loaded trained model')
		input.close()
	return classifier

def save_model(classifier):
	"""Saves the model"""
	print('saving trained model')
	with open('data/model.pkl', 'wb') as output:
		pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
		print('saved trained model')

def load_features():
	"""
	Loads the extracted features for each data set

	Returns:
		train_features  a dictionary of the features in the train set
		valid_features  a dictionary of the features in the validation set
		test_features   a dictionary of the features in the test set
	"""
	print('loading extracted features')
	train_features = np.load('data/train_features.npy')
	valid_features = np.load('data/valid_features.npy')
	test_features  = np.load('data/test_features.npy')
	print('loaded extracted features')
	return train_features, valid_features, test_features

def save_features(train_features, valid_features, test_features):
	"""Saves the extracted features for each dataset"""
	print('saving extracted features')
	np.save('data/train_features.npy', train_features)
	np.save('data/valid_features.npy', valid_features)
	np.save('data/test_features.npy', test_features)
	print('saved extracted features')

def build_vocabulary(inputs):
	"""
	Builds a dictionary of unique words in the corpus

	Returns:
		vocabulary  a dictionary of all the unique words in the corpus
	"""
	print('building vocabulary of words in the corpus')
	global vocabulary

	for tweet in inputs:
		for word in str(tweet).split():
			if vocabulary.has_key(word):
				vocabulary[word] += 1
			else:
				vocabulary[word] = 1

	print('built vocabulary of words in the corpus')
	return vocabulary

def build_features(document, i, vocabulary_words):
	if i % 10000 == 0:
		print('extracted features for {0} tweets'.format(i))

	document_words = set(str(document).split())
	features = np.zeros(len(vocabulary_words))
	for i in range(len(vocabulary_words)):
		features[i] = (vocabulary_words[i] in document_words)
	return features

def extract_features(inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test):
	"""
	Extracts features for training the model.

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
	vocabulary = build_vocabulary(inputs)

	# Get most common words from vocabulary
	global NUM_FEATURES
	words = dict(sorted(vocabulary.iteritems(), key=operator.itemgetter(1), reverse=True)[:NUM_FEATURES])
	words = words.keys()

	print('extracting features for all tweets')
	train_features = [(build_features(inputs_train[i], i, words)) for i in range(len(inputs_train))]
	valid_features = [(build_features(inputs_valid[i], i, words)) for i in range(len(inputs_valid))]
	test_features  = [(build_features(inputs_test[i], i, words)) for i in range(len(inputs_test))]
	print('extracted features for all tweets')

	return np.array(train_features), np.array(valid_features), np.array(test_features)

def train_model(features, targets):
	"""
	Trains a Naive Bayes classifier using the features passed in.

	Returns:
		classifier  the trained model
	"""
	print('training model')
	classifier = MultinomialNB()
	classifier.fit(features, targets)
	print('trained model')
	return classifier

def main():
	"""
	CLI Arguments allowed:
		display_graphs  Displays graphs
		retrain         Trains a new model
	"""

	inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test = load_parsed_data()

	if 'display_graphs' in sys.argv:
		display_graphs = True

	print('using {} percent of all data in corpus'.format(PERCENTAGE_DATA_SET_TO_USE*100))
	print('using {} most common words as features'.format(NUM_FEATURES))

	if not trained_model_exists() or '--retrain' in sys.argv:
		train_features, valid_features, test_features = extract_features(
			inputs_train[:len(inputs_train)*PERCENTAGE_DATA_SET_TO_USE],
			targets_train[:len(targets_train)*PERCENTAGE_DATA_SET_TO_USE],
			inputs_valid[:len(inputs_valid)*PERCENTAGE_DATA_SET_TO_USE],
			targets_valid[:len(targets_valid)*PERCENTAGE_DATA_SET_TO_USE],
			inputs_test[:len(inputs_test)*PERCENTAGE_DATA_SET_TO_USE],
			targets_test[:len(targets_test)*PERCENTAGE_DATA_SET_TO_USE]
		)

		save_features(train_features, valid_features, test_features)

		print train_features.shape
		print targets_train.shape
		print targets_train[0].shape
		print targets_train
		assert train_features.shape[0] == targets_train[:len(targets_train)*PERCENTAGE_DATA_SET_TO_USE].shape[0]
		classifier = train_model(train_features, targets_train[:len(targets_train)*PERCENTAGE_DATA_SET_TO_USE])
		save_model(classifier)

	else:
		train_features, valid_features, test_features = load_features()
		classifier = load_trained_model()

	perdictions = classifier.predict(test_features)
	print("Number of mislabeled points out of a total %d points : %d"
	% (test_features.shape[0],(targets_test[:len(targets_test)*PERCENTAGE_DATA_SET_TO_USE] != perdictions).sum()))

if __name__ == "__main__": main()
