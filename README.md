# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GURUMURTHY S
RegisterNumber: 212223230066
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
data.head:

![Screenshot 2024-10-24 091856](https://github.com/user-attachments/assets/3a8b6c97-5815-4e17-b3ea-c60addb3b8f6)

data.info:

![Screenshot 2024-10-24 091904](https://github.com/user-attachments/assets/19b05105-353a-487d-8da9-c688a558e59f)

data.isnull:

![Screenshot 2024-10-24 091911](https://github.com/user-attachments/assets/72789478-43b0-4d31-9f08-d0f7d1c3dab6)

Accuracy Score:

![image](https://github.com/user-attachments/assets/e63a3779-3db2-4d4e-a5ea-9471626002b1)

Confusion matric :

![image](https://github.com/user-attachments/assets/b9ee6f31-71f4-4481-bbe3-a5cd566bf394)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
