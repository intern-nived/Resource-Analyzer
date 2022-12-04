#Loading Libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

#Load DataSet
DataSet = pd.read_csv('./Datasets/UpdatedResumeDataSet.csv' ,encoding='utf-8')

#Data Preprocessing
def cleanDS(Text):
    Text = re.sub('httpS+s*', ' ', Text)  #removing URLs
    Text = re.sub('RT|cc', ' ', Text)  #removing RT and cc
    Text = re.sub('#S+', '', Text)  #removing hashtags
    Text = re.sub('@S+', '  ', Text)  #removing mentions
    Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', Text)  #removing punctuations
    Text = re.sub(r'[^x00-x7f]',r' ', Text) 
    Text = re.sub('s+', ' ', Text)  #removing extra whitespace
    return Text
DataSet['cleaned'] = DataSet.Resume.apply(lambda x: cleanDS(x))

#Generating Labels
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    DataSet[i] = le.fit_transform(DataSet[i])
#Saving the Labels
np.save('./buildData/classes.npy', le.classes_)    

cleanedText = DataSet['cleaned'].values
Target = DataSet['Category'].values
#Vectorizing and extracting features
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(cleanedText)
#saving the vectorizer
with open("./buildData/word_vectorizer.pickle", "wb") as file:
    pickle.dump(word_vectorizer, file)
WordFeatures = word_vectorizer.transform(cleanedText)


#Model Building
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,Target,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

#evalute metrics 
prediction = clf.predict(X_test)
print(metrics.classification_report(y_test, prediction))

#save the model
with open("./buildData/model.pickle", "wb") as file:
    pickle.dump(clf, file)

