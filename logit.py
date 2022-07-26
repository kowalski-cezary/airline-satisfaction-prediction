import pandas as pd

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

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

import statsmodels.api as sm
logit_model=sm.Logit(Y_train,X_train)
log=logit_model.fit()
print(log.summary2())

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, y_pred)
print(confusion_matrix)
