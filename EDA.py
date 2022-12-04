import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load DataSet
resumeDataSet = pd.read_csv('./Datasets/UpdatedResumeDataSet.csv' ,encoding='utf-8')
googleDataSet = pd.read_csv('./Datasets/google_job_skills.csv' ,encoding='utf-8')
amazonDataSet = pd.read_csv('./Datasets/amazon_jobs_dataset.csv' ,encoding='utf-8')

#Exploratory Data Analysis (EDA) of RESUME dataset
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=resumeDataSet)
plt.savefig('./Datasets/charts/RESUME_category_details.png')
#Generate Pie Chart
targetCounts = resumeDataSet['Category'].value_counts().reset_index()['Category']
targetLabels  = resumeDataSet['Category'].value_counts().reset_index()['index']
plt.figure( figsize=(25,25))
plt.suptitle('RESUME CATEGORY DISTRIBUTION')
source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
plt.savefig('./Datasets/charts/RESUME_category_details_pie.png')

#Exploratory Data Analysis (EDA) of Google Job skills dataset
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=googleDataSet)
plt.savefig('./Datasets/charts/GOOGLE_category_details.png')
#Generate Pie Chart
targetCounts = googleDataSet['Category'].value_counts().reset_index()['Category']
targetLabels  = googleDataSet['Category'].value_counts().reset_index()['index']
plt.figure( figsize=(25,25))
plt.suptitle('GOOGLE CATEGORY DISTRIBUTION')
source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
plt.savefig('./Datasets/charts/GOOGLE_category_details_pie.png')

#Exploratory Data Analysis (EDA) of Amazon Jobs dataset
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)
sns.countplot(y="Title", data=amazonDataSet)
plt.savefig('./Datasets/charts/AMAZON_category_details.png')
#Generate Pie Chart
targetCounts = amazonDataSet['Title'].value_counts().reset_index()['Title']
targetLabels  = amazonDataSet['Title'].value_counts().reset_index()['index']
plt.figure( figsize=(25,25))
plt.suptitle('AMAZON CATEGORY DISTRIBUTION')
source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
plt.savefig('./Datasets/charts/AMAZON_category_details_pie.png')