"""
Author: Ramaneek Gill

This program uses Multinomial Naive Bayes to predict tweet sentiments.

By default this trains on only 10% of the available dataset so that
old machines or laptops don't run into 12+ GB RAM usage.

Also by default this program only uses the top 2000 most common words
as features to save computation, setting this to a higher threshold
should improve accuracy.

For more ways on how to make this machine learning implementation
more powerful take a look at the constants in this program.

CLI Arguments allowed:
	give_me_the_data     Uses the full training set
	display_graphs       Displays graphs
	retrain              Trains a new model
	cross-validate       Runs cross validation to fine tune the model
	test-validation_set  Tests the latest trained model against the validation set
	test-test_set        Tests the latets trained model against the test set

 _____            _   _                      _      ___              _           _
/  ___|          | | (_)                    | |    / _ \            | |         (_)
\ `--.  ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_  / /_\ \_ __   __ _| |_   _ ___ _ ___
 `--. \/ _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __| |  _  | '_ \ / _` | | | | / __| / __|
/\__/ /  __/ | | | |_| | | | | | |  __/ | | | |_  | | | | | | | (_| | | |_| \__ \ \__ \\
\____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__| \_| |_/_| |_|\__,_|_|\__, |___/_|___/
                                                                        __/ |
                                                                       |___/

Machine learning has begun!
"""

import sys
import pickle
import os.path
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform as sp_rand
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

### Global variables
vocabulary = {} # A dictionary of all the unique words in the corpus

### Change me to higher values for better accuracy!
NUM_FEATURES = 2000 # The number of most common words in the corpus to use as features
PERCENTAGE_DATA_SET_TO_USE = 0.1 # The percentage of the dataset to use
N_CV_ITERS = 10 # The number of iterations to use in randomized cross validation

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

def save_model(classifier, prefix=''):
	"""Saves the model"""
	print('saving trained model')
	with open('data/'+prefix+'model.pkl', 'wb') as output:
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

def train_model(features, targets, alpha=1):
	"""
	Trains a Naive Bayes classifier using the features passed in.

	Returns:
		classifier  the trained model
	"""
	print('training model')
	classifier = MultinomialNB(alpha=alpha)
	classifier.fit(features, targets)
	print('trained model')
	return classifier

def cross_validate(train_features, targets_train, iters):
	"""
	Runs randomized cross validation using adjustable MultinomialNB params.

	Returns:
		The model that is the most accurate
	"""
	print('starting cross validation')
	param_grid = {'alpha': sp_rand()}
	model = MultinomialNB()
	rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=iters)
	rsearch.fit(train_features, targets_train)
	print('finished cross validation')
	print('best model has a score of {} using alpha={}'.format(rsearch.best_score_, rsearch.best_estimator_.alpha))
	return rsearch.best_estimator_.alpha

def plot_precision_and_recall(predictions, targets):
	"""Calculates and displays the precision and recall graph"""

	# Compute Precision-Recall and plot curve
	precision = dict()
	recall = dict()
	average_precision = dict()
	average_precision = average_precision_score(targets, predictions)
	precision, recall, _ = precision_recall_curve(targets, predictions)

	# Plot Precision-Recall curve
	plt.clf()
	plt.plot(recall, precision, label='Precision-Recall curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
	plt.legend(loc="lower left")
	plt.show()

def main():
	"""
	CLI Arguments allowed:
		--display_graphs       Displays graphs
		--retrain              Trains a new model
		--cross-validate       Runs cross validation to fine tune the model
		--test=validation_set  Tests the latest trained model against the validation set
		--test=test_set        Tests the latets trained model against the test set
	"""

	print(__doc__)

	inputs_train, targets_train, inputs_valid, targets_valid, inputs_test, targets_test = load_parsed_data()

	# Limit the data used to make it possible to run on old machines
	if 'give_me_the_data' not in sys.argv:
		inputs_train  = inputs_train[:len(inputs_train)*PERCENTAGE_DATA_SET_TO_USE]
		targets_train = targets_train[:len(targets_train)*PERCENTAGE_DATA_SET_TO_USE]
		inputs_valid  = inputs_valid[:len(inputs_valid)*PERCENTAGE_DATA_SET_TO_USE]
		targets_valid = targets_valid[:len(targets_valid)*PERCENTAGE_DATA_SET_TO_USE]
		inputs_test   = inputs_test[:len(inputs_test)*PERCENTAGE_DATA_SET_TO_USE]
		targets_test  = targets_test[:len(targets_test)*PERCENTAGE_DATA_SET_TO_USE]
	else:
		print('WARNING: You are using the entire data set, this will consume 12+ GB of RAM')

	if '--display_graphs' in sys.argv:
		display_graphs = True
	else:
		display_graphs = False

	print('using {} percent of all data in corpus'.format(PERCENTAGE_DATA_SET_TO_USE*100))
	print('using {} most common words as features'.format(NUM_FEATURES))

	if not trained_model_exists() or '--retrain' in sys.argv:
		train_features, valid_features, test_features = extract_features(
			inputs_train,
			targets_train,
			inputs_valid,
			targets_valid,
			inputs_test,
			targets_test
		)

		save_features(train_features, valid_features, test_features)
		classifier = train_model(train_features, targets_train)
		save_model(classifier)

	else:
		train_features, valid_features, test_features = load_features()
		classifier = load_trained_model()

	if '--cross-validate' in sys.argv:
		alpha = cross_validate(train_features, targets_train, N_CV_ITERS)
		classifier = train_model(train_features, targets_train, alpha)
		save_model(classifier, 'cross_validated_')

	if '--test=validation_set' in sys.argv:
		score = classifier.score(valid_features, targets_valid)
		print('Accuracy against validation set is {} percent'.format(score*100))

		if display_graphs == True:
			predictions = classifier.predict(valid_features)
			plot_precision_and_recall(predictions, targets_valid)

	if '--test=test_set' in sys.argv:
		score = classifier.score(test_features, targets_test)
		print('Accuracy against test set is {} percent'.format(score*100))

		if display_graphs == True:
			predictions = classifier.predict(test_features)
			plot_precision_and_recall(predictions, targets_test)

if __name__ == "__main__": main()
