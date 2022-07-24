import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

df_train = pd.read_csv("C:/Users/Czarek/Documents/train.csv")
df_test = pd.read_csv("C:/Users/Czarek/Documents/test.csv")

df = pd.concat([df_train, df_test])
df.dropna(axis=0, inplace=True)
df.drop('id', axis=1, inplace=True)

Gender = {'Male':0,
        'Female':1}

# apply using map
df['Gender'] = df['Gender'].map(Gender)

CustomerType= {'Loyal Customer':0,'disloyal Customer':1}

# apply using map
df['Customer Type'] = df['Customer Type'].map(CustomerType)

TypeofTravel= {'Business travel':0,'Personal Travel':1}

# apply using map
df['Type of Travel'] = df['Type of Travel'].map(TypeofTravel)

Class= {'Business':0,'Eco':1,'Eco Plus':2}

# apply using map
df['Class'] = df['Class'].map(Class)

satisfaction= {'neutral or dissatisfied':0,'satisfied':1}

# apply using map
df['satisfaction'] = df['satisfaction'].map(satisfaction)

X = df.iloc[:, 0:23].values
Y = df.iloc[:,23].values

X = pd.DataFrame(X)
X.head()

Y = pd.DataFrame(Y)
Y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
Tree1 =DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=1000,
                              random_state = 0)
print(Tree1.fit(X_train, Y_train))

fig = plt.figure(figsize=(17,8))
_ = tree.plot_tree(Tree1,
                   class_names= ('Dissatisfied/Neutral', 'Satisfied'),
                   feature_names= df.columns,
                   filled=True)
plt.show()

#Check Accuracy precision, recall, f1-score
print( classification_report(Y_test, Tree1.predict(X_test)) )
#Another way to get the models accuracy on the test data
print(F'Accuracy:',accuracy_score(Y_test, Tree1.predict(X_test)))
print(F'Precision:', precision_score(Y_test, Tree1.predict(X_test)))
print(F'Recall:', recall_score(Y_test, Tree1.predict(X_test)))
print(F'F1 Score:', f1_score(Y_test, Tree1.predict(X_test)))

#Check Roc Auc Score
print( F'Roc Auc Score:',roc_auc_score(Y_test, Tree1.predict(X_test)) )
print( F'Balanced Accuracy Score:',balanced_accuracy_score(Y_test, Tree1.predict(X_test)) )
print( F'Confusion Matrix:',confusion_matrix(Y_test, Tree1.predict(X_test)) )
print()#Print a new line

# ROC CURVE
plot_roc_curve(Tree1, X_test, Y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
