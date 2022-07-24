import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline

data = pd.read_csv("C:/Users/Czarek/Documents/train.csv")
data.shape

data = data.drop(data.iloc[:,[0, 1]], axis = 1)

categorical_indexes = [0, 1, 3, 4] + list(range(6, 20))
data.iloc[:,categorical_indexes] = data.iloc[:,categorical_indexes].astype('category')


f, ax = plt.subplots(1, 2, figsize = (20,5))
sns.countplot(x = 'Seat comfort', hue = 'satisfaction', palette = "YlOrBr", data = data,ax = ax[0])
sns.countplot(x = 'Leg room service', hue = 'satisfaction', palette = "YlOrBr", data = data, ax = ax[1])
plt.show()

plt.pie(data.satisfaction.value_counts(), labels = ["Neutral or dissatisfied", "Satisfied"], colors = sns.color_palette("YlOrBr"), autopct = '%1.1f%%')
pass

categ = data.iloc[:,categorical_indexes]
fig, axes = plt.subplots(6, 3, figsize = (30, 30))
for i, col in enumerate(categ):
    column_values = data[col].value_counts()
    labels = column_values.index
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 50}
    sizes = column_values.values
    axes[i//3, i%3].pie(sizes, labels = labels, colors = sns.color_palette("YlOrBr"), autopct = '%1.0f%%', startangle = 90)
    axes[i//3, i%3].axis('equal')
    axes[i//3, i%3].set_title(col)
plt.show()

f, ax = plt.subplots(1, 2, figsize = (20,5))
sns.histplot(x = "Departure Delay in Minutes", stat='probability', palette = "YlOrBr", data = data, bins=100, binrange=(0,150),ax = ax[0])
sns.histplot(x = "Arrival Delay in Minutes", stat='probability', palette = "YlOrBr", data = data,bins=100, binrange=(0,150),ax = ax[1])
plt.show()
f, ax = plt.subplots(1, 2, figsize = (20,5))
sns.histplot(x = "Age",  palette = "YlOrBr",stat='probability', data = data,ax = ax[0])
sns.histplot(x = "Flight Distance",  palette = "YlOrBr", stat='probability',data = data,ax = ax[1])
plt.show()


sns.countplot(x = 'Class', hue = 'satisfaction', palette = "YlOrBr", data = data)
plt.show()
sns.countplot(x = 'Type of Travel', hue = 'satisfaction', palette = "YlOrBr", data = data)
plt.show()
sns.countplot(x = 'Online boarding', hue = 'satisfaction', palette = "YlOrBr", data = data)
plt.show()

sns.countplot(x = 'Age', palette = "YlOrBr", data = data)
plt.show()

# Fill in the missing values with <b>medians</b> in the columns corresponding to quantitative features:

data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].median(axis = 0), inplace = True)


numerical_columns = [c for c in data.columns if data[c].dtype.name != 'category']
numerical_columns.remove('satisfaction')
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'category']
data_describe = data.describe(include = ['category'])

binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)

for col in binary_columns:
    print(col, ': ', end = '')
    for uniq in data[col].unique():
        if uniq == data[col].unique()[-1]:
            print(uniq, end = '.')
        else:
            print(uniq, end = ', ')
    print()

for col in binary_columns:
    data[col] = data[col].astype('object')
    k = 0
    for uniq in data[col].unique():
        data.at[data[col] == uniq, col] = k
        k +=1
for col in binary_columns:
    print(data[col].describe(), end = '\n\n')

data[nonbinary_columns]

data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)

len(data_nonbinary.columns)

data_numerical = data[numerical_columns]
data_numerical.describe()

data_numerical = (data_numerical - data_numerical.mean(axis = 0))/data_numerical.std(axis = 0)

data_numerical.describe()

target = data['satisfaction']
data = pd.concat((data_numerical, data_nonbinary, data[binary_columns]), axis = 1)
print(data.shape)

data.describe()

X = data
y = target
N, d = X.shape
N, d


