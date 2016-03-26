import csv
import sys
import numpy as np

STOP_WORDS = []

def load_stopwords():
	"""Loads the stopwords.txt file into an array"""

	print('loading stopwords')
	with open('stopwords.txt') as f:
		global STOP_WORDS
		STOP_WORDS = f.read().splitlines()
	print('loaded stopwords')


def main():
	cli_args = sys.argv

	if 'remove_stopwords' in cli_args:
		load_stopwords()

	load_csv()



if __name__ == "__main__": main()
