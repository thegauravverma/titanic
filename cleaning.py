import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
train=pd.read_csv("dataset/train.csv")
test=pd.read_csv("dataset/test.csv")
submit=pd.read_csv("dataset/submission.csv")
test['Survived']=submit['Survived']
features=['Age','Sex','Pclass','Embarked']
feature=pd.concat([train[features],test[features]])
label=pd.concat([train['Survived'],test['Survived']])
feature['Survived']=label
feature['Age']=feature['Age'].fillna(feature['Age'].mean())
feature=feature.dropna()
lr=LabelEncoder()
#0 for female -1 for male
feature['Sex']=lr.fit_transform(feature['Sex'])
print(lr.classes_)
#C-0  Q-1  S-2
feature['Embarked']=lr.fit_transform(feature['Embarked'])
print(lr.classes_)
feature.to_csv('final.csv')