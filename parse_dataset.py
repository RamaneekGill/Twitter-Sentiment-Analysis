import sys
import numpy as np
from pandas import read_csv

STOP_WORDS = []
FILENAME = 'training_data.csv' # the training data csv

def load_stopwords():
	"""Loads the stopwords.txt file into an array"""
	print('loading stopwords')
	with open('stopwords.txt') as f:
		global STOP_WORDS
		STOP_WORDS = f.read().splitlines()
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

	print('Removed {0} neutral tweets'.format(count))
	return inputs, targets

def main():
	cli_args = sys.argv

	if 'remove_stopwords' in cli_args:
		load_stopwords()

	corpus = load_csv()
	inputs, targets = parse_corpus(corpus)

	if not 'keep_neutral_tweets' in cli_args:
		inputs, targets = remove_neutral_tweets(inputs, targets)

	exit()



if __name__ == "__main__": main()
