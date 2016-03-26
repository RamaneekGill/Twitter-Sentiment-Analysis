# Twitter-Sentiment-Analysis
Contains machine learning methods on predicting sentiments for tweets.

### Data Used:
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
 - scikit-learn
 - TensorFlow

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
in order to determine if the independance assumed from a Bayesian Network has an
advantage or disadvantage over a Gaussian class-conditional density in an LDA model
for short texts.

### References

Li, F., Huang, M., & Zhu, X. (2010, July). Sentiment Analysis with Global Topics and Local Dependency. In AAAI (Vol. 10, pp. 1371-1376).
https://www.cs.princeton.edu/~blei/papers/Blei2011.pdf

Bai, X. (2011). Predicting consumer sentiments from online text. Decision Support Systems, 50(4), 732-742.
http://www.sciencedirect.com/science/article/pii/S016792361000148X

Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1, 12.
http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf