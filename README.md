# Twitter-Sentiment-Analysis
Contains machine learning methods used for predicting sentiments for tweets. :8ball: :bird: !

### How to run this code yourself:
- This repo assumes you have the tools I used installed.
 - Visit the [Tools Used](### Tools Used) section


- Clone this repo. For convenience the parsed dataset has been uploaded in a numpy array format
 - To parse the raw data yourself if you've made changes to the `parse_dataset.py` file you will need the raw dataset, source for the raw data is below in this readme.
 - Run `python parse_dataset.py`
 - There are some convenient arguments you can pass in, a list of them are included in the doc block for the main method in `parse_dataset.py`


- Run `python *_runner.py` in the folder for the machine learning algorithm you wish to run
 - e.g. `python *_runner.py --display_graphs`
 - I may or may not have uploaded the trained models into this repo. If there are no saved models/weights the scripts will automatically retrain.
 - Please don't be mad if you have to train most models yourself, disk space on a small SSD is hard to come by
 - If you really don't want to train the models yourself but are super curious shoot me an email!


- To (re)train the models run `python *_runner.py --retrain`
 - To run cross validation to train the best fine tuned model run `python *_runner.py --retrain --cross-validate`


- To display the graphs run `python *_runner.py --display_graphs`


- To see the accuracy results for a model run `python *_runner.py --test=test_set`
 - No I did not cheat while training, commit history will show everything I've done.


- To use all of the data run `python *_runner.py --give_me_the_data`
 - WARNING: this will easily consume 16+ GB of RAM for most of the models

### Data Source Used:
- http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
 - ~1.6 million tweet dataset with positive/negative sentiment attached

### Machine Learning Methods Used:
 - Naive Bayesian Network
 - LDA (Latent Dirichlet Allocation)
 - TODO: Linear Regression
 - TODO: SVM
 - TODO: Logistic Regression
 - TODO: Neural Networks

### Tools Used:
 - NumPy
 - sklearn
 - TensorFlow
 - NLTK
 - Install these by running `pip install {package name}` in your terminal/shell

### Inquiries:
Please email me at ramaneek.gill@hotmail.com

### Research Paper Proposal
We plan on creating a short text sentiment analyzer. This project will be using a dataset
of ~1.6 million tweets (from: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
that contains tweet text from twitter along with a boolean flag showing whether the tweet
is positive or not.

We will be training two models on this data. One will be trained using a Bayesian Network,
the other will be trained using Latent Dirichlet allocation (LDA). The two models'
classification errors, precision, and recall will be compared against each other
in order to determine if the independence assumed from a Bayesian Network has an
advantage or disadvantage over a Gaussian class-conditional density in an LDA model
for short texts.

### References

Li, F., Huang, M., & Zhu, X. (2010, July). Sentiment Analysis with Global Topics and Local Dependency. In AAAI (Vol. 10, pp. 1371-1376).
https://www.cs.princeton.edu/~blei/papers/Blei2011.pdf

Bai, X. (2011). Predicting consumer sentiments from online text. Decision Support Systems, 50(4), 732-742.
http://www.sciencedirect.com/science/article/pii/S016792361000148X

Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1, 12.
http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf
