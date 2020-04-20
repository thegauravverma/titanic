import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
df=pd.read_csv("final.csv")
X=df[['Age','Sex','Pclass','Embarked']]
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=25,stratify=y)
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))
pickle.dump(rfc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))
pred=model.predict([[43,0,1,0]])
print(pred)