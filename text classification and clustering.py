#Text classification
#reference: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

#if download data set is from library
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


from sklearn.datasets import load_files

#if downloaded the data set already
twenty_train = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-train',
    encoding='latin-1', random_state=42)
twenty_test = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-test',
    encoding='latin-1', random_state=42)


#
#In order to run machine learning algorithms we need to convert the text files into numerical feature vectors.Each unique word in our dictionary will correspond to a feature (descriptive feature).
#We will be using bag of words model for our example.
#Briefly, we segment each text file into words (for English splitting by space), and count
#
#Here by doing ‘count_vect.fit_transform(twenty_train.data)’, we are learning the vocabulary dictionary and it returns a Document-Term matrix. [n_samples, n_features].
#
#TF: Just counting the number of words in each document has 1 issue: it will give more weightage to longer documents than shorter documents. To avoid this, we can use frequency (TF - Term Frequencies) i.e. #count(word) / #Total words, in each document.
#
#TF-IDF: Finally, we can even reduce the weightage of more common words like (the, is, an etc.) which occurs in all document. This is called as TF-IDF i.e Term Frequency times inverse document frequency.
#



# of times each word occurs in each document and finally assign each word an integer id.
#Exxtract and Runn9ng algorithm for NB at once
##############################
#Extract features from text file
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
tfidf_transformer.get_feature_names()


##Running ML Algorithm for naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target_names)
##############################################
#Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
#Exxtract and Runn9ng algorithm for NB at once

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                  ('clf', MultinomialNB()), ])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#check performance of NB classifier with test set
import numpy as np
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)


#Exxtract and Runn9ng algorithm for SVM
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                       alpha=1e-3, n_iter=5, random_state=42)), ])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

#check the  SVM algorithm with test data
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)


#if csv file
import pandas as pd
data = pd.read_csv(‘your.csv’) #text in column 1, classifier in column 2.
import numpy as np
numpy_array = data.as_matrix()
X = numpy_array[:,0]             #x=numpy_array[:,:3]
Y = numpy_array[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
 X, Y, test_size=0.4, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([(‘vect’, CountVectorizer(stop_words=’english’)),
 (‘tfidf’, TfidfTransformer()),
 (‘clf’, MultinomialNB()),
])

text_clf = text_clf.fit(X_train,Y_train)

predicted = text_clf.predict(X_test)
np.mean(predicted == Y_test)

#reference:https://pythonprogramminglanguage.com/kmeans-text-clustering/
#Text clustering

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :20]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)
 
