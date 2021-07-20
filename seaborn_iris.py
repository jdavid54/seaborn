import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/index.html
# https://github.com/mwaskom/seaborn-data

# appel de fonction seaborn
# sns.fonction(x,y,data, hue,size,style)

## 1. Pairplot() : La vue d'ensemble
iris = sns.load_dataset('iris')
print(iris.head())
sns.pairplot(iris)
plt.show()

# avec hue
sns.pairplot(iris, hue='species')
plt.show()

## 2. Visualiser de catégories
titanic = sns.load_dataset('titanic')
titanic.drop(['alone', 'alive', 'who', 'adult_male', 'embark_town', 'class'], axis=1, inplace=True)
titanic.dropna(axis=0, inplace=True)
print(titanic.head())

sns.catplot(x='survived', y='age', data=titanic, hue='sex')
plt.show()

plt.figure(figsize=(32, 8))
sns.boxplot(x='age', y='fare', data=titanic, hue='sex')
plt.show()


## 3. Visualisation de Distributions
sns.distplot(titanic['fare'])
plt.show()


sns.jointplot('age', 'fare', data=titanic, kind='hex')
plt.show()

sns.heatmap(titanic.corr())
plt.show()