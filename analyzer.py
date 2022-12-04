#Loading Libraries
from PyPDF2 import PdfReader
import docxpy
import os
import pickle
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

#Get project folder path
print("SELECT PROJECT FOLDER:")
path = filedialog.askdirectory()

if(path == ''):
    print("INVALID PATH! try again...")
    quit()
else:
    print(".")
    print(".")
    print("SCANNING FILES...")
#Extarct Texts from all the documents present in the project folder
textData = []
for dirpath, subdirs, files in os.walk(path):
    for x in files:
        file = os.path.join(dirpath, x)

        if file.endswith(".pdf"): 
            reader = PdfReader(file)
            for page in reader.pages:
                textData.append(page.extract_text())      # Extarct Texts from PDF files

        if file.endswith(".docx"):
            textData.append(docxpy.process(file))         # Extarct Texts from Docx files
print(".")
print(".")
if(len(textData)==0):
    print("NO RELEVANT DATA FOUND! please try with texual data only...")
    quit()
else:    
    print("PROCESSING FILES...")
print(".")
print(".")
#Creating DataFrame for the Extracted Texts
df = pd.DataFrame(textData, columns=['ExtractedText'])


#Load labels used by model as Text
model_labels = np.load('./buildData/classes.npy', allow_pickle=True)
#Load trained model
model = pickle.load(open("./buildData/model.pickle", "rb"))

#Data Preprocessing
def cleanDF(ctext):
    ctext = re.sub('httpS+s*', ' ', ctext)  # remove URLs
    ctext = re.sub('RT|cc', ' ', ctext)  # remove RT and cc
    ctext = re.sub('#S+', '', ctext)  # remove hashtags
    ctext = re.sub('@S+', '  ', ctext)  # remove mentions
    ctext = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', ctext)  # remove punctuations
    ctext = re.sub(r'[^x00-x7f]',r' ', ctext) 
    ctext = re.sub('s+', ' ', ctext)  # remove extra whitespace
    return ctext

df['cleaned'] = df.ExtractedText.apply(lambda x: cleanDF(x))
cleanedText = df['cleaned'].values

#Vectorizing the Texts
word_vectorizer = pickle.load(open("./buildData/word_vectorizer.pickle", "rb"))
WordFeatures = word_vectorizer.transform(cleanedText)

#Prediction
prediction = model.predict(WordFeatures)

pielist= []
for i in prediction:
    pielist.append(model_labels[i])         #Converting predicted results to text results


#Generating Pie Chart
targetCounts =[]
lablelist= sorted(set(pielist))
for j in lablelist:
    targetCounts.append(pielist.count(j))   

plt.figure(1, figsize=(10,5))
source_pie = plt.pie(targetCounts, labels=lablelist, autopct=lambda pct: '{:1.1f}%'.format(pct) if pct >0 else '', shadow=True)
plt.tight_layout()
plt.suptitle("Project Resource Analysis:")
plt.savefig('./output.png')
plt.show()
print("OUTPUT SAVED...")
