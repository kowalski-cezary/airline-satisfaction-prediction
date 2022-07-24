import pandas as pd
import matplotlib.pyplot as plt
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

df_test.shape
print(df_test.shape)


df = pd.concat([df_train, df_test])
df.dropna(axis=0, inplace=True)
df.drop('id', axis=1, inplace=True)

Gender = {'Male':0,
        'Female':1}

df['Gender'] = df['Gender'].map(Gender)

CustomerType= {'Loyal Customer':0,'disloyal Customer':1}

df['Customer Type'] = df['Customer Type'].map(CustomerType)

TypeofTravel= {'Business travel':0,'Personal Travel':1}

df['Type of Travel'] = df['Type of Travel'].map(TypeofTravel)

Class= {'Business':0,'Eco':1,'Eco Plus':2}

df['Class'] = df['Class'].map(Class)

satisfaction= {'neutral or dissatisfied':0,'satisfied':1}

df['satisfaction'] = df['satisfaction'].map(satisfaction)

X = df.iloc[:, 0:23].values
Y = df.iloc[:,23].values

X = pd.DataFrame(X)
X.head()

Y = pd.DataFrame(Y)
Y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
print(forest.fit(X_train, Y_train))
print('[0]Logistic Regression Training Accuracy:', forest.score(X_train, Y_train))

#Check Accuracy precision, recall, f1-score
print( classification_report(Y_test, forest.predict(X_test)) )

#Another way to get the models accuracy on the test data
print(F'Accuracy:',accuracy_score(Y_test, forest.predict(X_test)))
print(F'Precision:', precision_score(Y_test, forest.predict(X_test)))
print(F'Recall:', recall_score(Y_test, forest.predict(X_test)))
print(F'F1 Score:', f1_score(Y_test, forest.predict(X_test)))

#Check Roc Auc Score
print( F'Roc Auc Score:',roc_auc_score(Y_test, forest.predict(X_test)) )
print( F'Balanced Accuracy Score:',balanced_accuracy_score(Y_test, forest.predict(X_test)) )
print( F'Confusion Matrix:',confusion_matrix(Y_test, forest.predict(X_test)) )
print()#Print a new line

# ROC CURVE
plot_roc_curve(forest, X_test, Y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()