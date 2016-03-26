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
	corpus.columns = ["sentiment", "2", "3", "4", "5", "tweet"]
	inputs = np.array(corpus["tweet"])
	targets = np.array(corpus["sentiment"])

	return inputs, targets

def main():
	cli_args = sys.argv

	if 'remove_stopwords' in cli_args:
		load_stopwords()

	corpus = load_csv()
	inputs, targets = parse_corpus(corpus)
	exit()



if __name__ == "__main__": main()
