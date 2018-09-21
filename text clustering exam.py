%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray()


# Load the text data
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
twenty_train_small = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-train',
    categories=categories, encoding='latin-1')
twenty_test_small = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-test',
    categories=categories, encoding='latin-1')

#skitch learn matrix
vectorizer = TfidfVectorizer(min_df=1)
%time X_train_small = vectorizer.fit_transform(twenty_train_small.data)
type(vectorizer.vocabulary_)
len(vectorizer.vocabulary_)
len(vectorizer.get_feature_names())
vectorizer.get_feature_names()[:10]
vectorizer.get_feature_names()[(n_features / 2):(n_features / 2 + 10)]

#scipy.sparse matrix.
#converted to matrix format.nsamples(tokens) is row.n_features(individual token occurences) is column.
n_samples, n_features = X_train_small.shape


# Turn the text documents into vectors of word frequencies
vectorizer = TfidfVectorizer(min_df=2)
X_train = vectorizer.fit_transform(twenty_train_small.data)
y_train = twenty_train_small.target

# Fit a classifier on the training set
classifier = MultinomialNB().fit(X_train, y_train)
print("Training score: {0:.1f}%".format(
    classifier.score(X_train, y_train) * 100))

# Evaluate the classifier on the testing set
X_test = vectorizer.transform(twenty_test_small.data)
y_test = twenty_test_small.target
print("Testing score: {0:.1f}%".format(
    classifier.score(X_test, y_test) * 100))

#The load_files function can load text files from a 2 levels folder structure assuming folder names represent categories:

all_twenty_train = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-train',
    encoding='latin-1', random_state=42)
all_twenty_test = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-test',
    encoding='latin-1', random_state=42)

all_target_names = all_twenty_train.target_names
all_twenty_train.target
all_twenty_train.target.shape
all_twenty_test.target.shape
len(all_twenty_train.data)
len(all_twenty_test.data)
type(all_twenty_train.data[0])

#display class name amd data of data set.
def display_sample(i, dataset):
    print("Class name: " + dataset.target_names[dataset.target[i]])
    print("Text content:\n")
    print(dataset.data[i])
    
display_sample(1, all_twenty_train)

#Let's compute the (uncompressed, in-memory) size of the training and test sets in MB assuming an 8-bit encoding (in this case, all chars can be encoded using the latin-1 charset).
def text_size(text, charset='iso-8859-1'):
    return len(text.encode(charset)) * 8 * 1e-6

train_size_mb = sum(text_size(text) for text in all_twenty_train.data) 
test_size_mb = sum(text_size(text) for text in all_twenty_test.data)

print("Training set size: {0} MB".format(int(train_size_mb)))
print("Testing set size: {0} MB".format(int(test_size_mb)))

