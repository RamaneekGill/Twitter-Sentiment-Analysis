import sys
import string
import numpy as np
from pandas import read_csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

STOP_WORDS = np.array([])
FILENAME = 'training_data.csv' # the training data csv
RANDOM_SEED = 1337

def load_stopwords():
	"""Loads the stopwords.txt file into an array"""
	print('loading stopwords')
	with open('stopwords.txt') as f:
		global STOP_WORDS
		STOP_WORDS = np.array(f.read().splitlines())
	print('loaded stopwords')

def load_csv():
	"""
	Loads the csv file

	Returns:
		corpus: An array version of the FILENAME
	"""
	print('reading from csv')
	corpus = read_csv(FILENAME)
	print('read from csv')
	return corpus

def parse_corpus(corpus):
	"""
	Parses the corpus and returns the inputs and targets

	Returns:
		inputs:  A numpy array of the tweets
		targets: A numpy array of the sentiment, 1 for positive, 0 for negative
	"""
	print('parsing corpus')
	corpus.columns = ["sentiment", "2", "3", "4", "5", "tweet"]
	inputs = np.array(corpus["tweet"])
	targets = np.array(corpus["sentiment"])
	print('parsed corpus into numpy arrays')

	return inputs, targets

def remove_neutral_tweets(inputs, targets):
	"""
	Parses the corpus and returns the inputs and targets

	Returns:
		inputs:  A numpy array of the tweets
		targets: A numpy array of the sentiment, 1 for positive, 0 for negative
	"""
	print('removing neutral tweets')

	count = 0
	for i in range(len(inputs)):
		if targets[i] == 2: # Remove tweets with neutral sentiment
			count += 1
			np.delete(inputs, i)
			np.delete(targets, i)

	print('removed {0} neutral tweets'.format(count))
	return inputs, targets

def remove_stopwords(inputs, stopwords):
	"""
	Parses the inputs and removes stopwords.

	Returns:
		inputs:  A numpy array of the tweets
	"""
	print('removing stopwords from tweets')

	count = 0
	for i in range(len(inputs)):
		tweet_list = inputs[i].split()
		inputs[i] = ' '.join([j for j in tweet_list if j not in stopwords])

	print('removed stopwords from tweets')
	return inputs

def	remove_empty_tweets(inputs, targets):
	"""
	Parses the inputs and removes input and target where the input is empty.
	Removes data where the tweet content is empty.

	Returns:
		inputs:  A numpy array of the tweets
		targets: A numpy array of the sentiment, 1 for positive, 0 for negative
	"""
	print('removing empty tweets')

	i = 0
	count = 0
	length = len(inputs)
	while i < length:
		if inputs[i] == ' ' or inputs[i] == '':
			inputs = np.delete(inputs, i)
			targets = np.delete(targets, i)
			i -= 1
			length -= 1
		i += 1

	print('removed {0} tweets from dataset since tweets were empty'.format(count))
	return inputs, targets

def remove_punctuation(inputs):
	"""
	Parses the inputs and removes punctuation from tweet content.

	Returns:
		inputs:  A numpy array of the tweets
	"""
	print('removing punctuation from tweet content')
	table = string.maketrans("","")
	for i in range(len(inputs)):
		inputs[i] = inputs[i].translate(table, string.punctuation)
	print('removed punctuation from tweet content')

	return inputs

def main():
	"""
	CLI Arguments allowed:
		keep_stopwords       Keep stopwords in tweet content
		                     By default removes stopwords

		keep_neutral_tweets  Keeps tweets with neutral sentiment
		                     By default removes neutral tweets

		keep_punctuation     Keeps punctuation in tweet content
		                     By default removes punctuation from tweet content
	"""
	cli_args = sys.argv

	corpus = load_csv()
	inputs, targets = parse_corpus(corpus)
	raw_inputs = inputs
	targets = (targets > 0) * 1 # Changes target array to 0s and 1s

	if not 'keep_neutral_tweets' in cli_args:
		inputs, targets = remove_neutral_tweets(inputs, targets)

	if not 'keep_stopwords' in cli_args:
		load_stopwords()
		inputs = remove_stopwords(inputs, STOP_WORDS)

	if not 'keep_punctuation' in cli_args:
		inputs = remove_punctuation(inputs)

	inputs, targets = remove_empty_tweets(inputs, targets)

	# Train set size      = 80 %
	# Test set size       = 10 %
	# Validation set size = 10 %
	print('splitting data in to train, test, and validation sets')
	inputs_train, inputs_test, targets_train, targets_test = train_test_split(
		inputs,
		targets,
		test_size=0.2,
		random_state=RANDOM_SEED
	)

	inputs_test, inputs_valid, targets_test, targets_valid = train_test_split(
		inputs_test,
		targets_test,
		test_size=0.5,
		random_state=RANDOM_SEED
	)
	print('split data in to train, test, and validation sets')

	assert len(inputs_train) == len(targets_train)
	assert len(inputs_valid) == len(targets_valid)
	assert len(inputs_test)  == len(targets_test)

	print('Length of train set: {0}'.format(len(inputs_train)))
	print('Length of validation set: {0}'.format(len(inputs_valid)))
	print('Length of test set: {0}'.format(len(inputs_test)))

	total_data_length = len(inputs_train) + len(inputs_valid) + len(inputs_test)
	original_data_length = len(raw_inputs)

	print('Original dataset length: {0}, parsed dataset length: {1}'.format(
		original_data_length,
		total_data_length
	))

	print('saving parsed dataset')
	np.save('parsed_data/inputs_train.npy', inputs_train)
	np.save('parsed_data/targets_train.npy', targets_train)
	np.save('parsed_data/inputs_valid.npy', inputs_valid)
	np.save('parsed_data/targets_valid.npy', targets_valid)
	np.save('parsed_data/inputs_test.npy', inputs_test)
	np.save('parsed_data/targets_test.npy', targets_test)
	print('saved parsed dataset')

	print('parsing complete!')

if __name__ == "__main__": main()
