import nltk
import sys

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
	print('saving parsed dataset')
    inputs_train  = np.load('../parsed_data/inputs_train.npy')
    targets_train = np.load('../parsed_data/targets_train.npy')
    inputs_valid  = np.load('../parsed_data/inputs_valid.npy')
    targets_valid = np.load('../parsed_data/targets_valid.npy')
    inputs_test   = np.load('../parsed_data/inputs_test.npy')
    targets_test  = np.load('../parsed_data/targets_test.npy')

	return inputs_train, targets_train,
	       inputs_valid, targets_valid,
	       inputs_test, targets_test


def main():
	"""
	CLI Arguments allowed:
		display_graphs  Displays graphs
		retrain         Trains a new model
	"""

	inputs_train, targets_train,
	inputs_valid, targets_valid,
	inputs_test, targets_test = load_parsed_data()

	if !trained_model_exists() or 'retrain' in sys.argv:
		train_model()

	if 'display_graphs' in sys.argv:
		display_graph()

if __name__ == "__main__": main()
