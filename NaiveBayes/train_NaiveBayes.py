import sys
import nltk
import operator
import numpy as np

display_graphs = False # Boolean flag for displaying graphs
vocabulary = {} # A dictionary of all the unique words in the corpus
NUM_FEATURES = 500 # The number of most common words in the corpus to use as features

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

def load_trained_model():
	"""
	Loads and returns the trained model

	Returns:
		model  the model
	"""
	pass # TODO: implement later when saving features dictionary

def load_features():
	"""
	Loads the extracted features for each data set

	Returns:
		train_features  a dictionary of the features in the train set
		valid_features  a dictionary of the features in the validation set
		test_features   a dictionary of the features in the test set
	"""
	pass # TODO: implement later

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
	global vocabulary

	if i % 500 == 0:
		print('extracted features for {0} tweets'.format(i))

	document_words = set(str(document).split())
	features = {}
	for word in vocabulary_words:
		features['{}'.format(word)] = (word in document_words)
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

	print('extracting features')
	train_features = [(build_features(inputs_train[i], i, words), targets_train[i]) for i in range(len(inputs_train))]
	valid_features = [(build_features(inputs_valid[i], i, words), targets_valid[i]) for i in range(len(inputs_valid))]
	test_features  = [(build_features(inputs_test[i], i, words), targets_test[i]) for i in range(len(inputs_test))]
	print('extracted features')

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
		train_features, valid_features, test_features = extract_features(
			inputs_train[:len(inputs_train)*0.1],
			targets_train[:len(targets_train)*0.1],
			inputs_valid[:len(inputs_valid)*0.1],
			targets_valid[:len(targets_valid)*0.1],
			inputs_test[:len(inputs_test)*0.1],
			targets_test[:len(targets_test)*0.1]
		)
	else:
		train_features, valid_features, test_features = load_features()
		classifier = load_trained_model()

	print('training model')
	classifier = nltk.NaiveBayesClassifier.train(train_features)
	print('trained model')
	print(nltk.classify.accuracy(classifier, test_features))
	print(classifier.show_most_informative_features(50))

if __name__ == "__main__": main()
