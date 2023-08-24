
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
import os
os.chdir("C:/Users/yaraj/OneDrive/Desktop")
data = pd.read_csv("data.csv")
x_train,x_test,y_train, y_test = train_test_split(data.iloc[:,[3,4,5,6,7,8,9,10]],data.iloc[:,11],test_size=0.2,random_state=42)
gnb = GaussianNB ()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(y_pred)
y_pred=pd.DataFrame(y_pred)
print (y_pred.value_counts())
print ("Accuracy", metrics.accuracy_score (y_test,y_pred))





