#reference: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a


#text classification 
from sklearn.datasets import load_files

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

twenty_train = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-train',
    encoding='latin-1', random_state=42)
twenty_test = load_files('G:\\sliit DS\\3rd year 2nd seme\\IRWA\\python\\20news-bydate\\20news-bydate-test',
    encoding='latin-1', random_state=42)


twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:4])) #prints lines of the first data file

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


##Running ML Algorithm for naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
##############################################
#Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
#Exxtract and Runn9ng algorithm for NB at once

from sklearn.pipeline import Pipeline
 
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

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                       alpha=1e-3, n_iter=5, random_state=42)), ])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

#check the  SVM algorithm with test data
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)



#To turn obtain optimal performace in NB algorithm sklearn provides gridsearch tool
from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

gs_clf.best_score_
gs_clf.best_params_

#To turn obtain optimal performace in SVM algorithm sklearn provides gridsearch tool
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__alpha': (1e-2, 1e-3),
 }
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
gs_clf_svm.best_score_
gs_clf_svm.best_params_

#using stemming and stop words
import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
...                      ('tfidf', TfidfTransformer()),
...                      ('mnb', MultinomialNB(fit_prior=False)),
... ])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
np.mean(predicted_mnb_stemmed == twenty_test.target)







#if csv file
import pandas as pd
data = pd.read_csv(‘your.csv’) #text in column 1, classifier in column 2.
import numpy as np
numpy_array = data.as_matrix()
X = numpy_array[:,0]
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