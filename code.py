import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

df.drop(columns=["Cabin"], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'])

df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

df.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

sns.boxplot(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.title("Boxplot for Outlier Detection")
plt.show()

z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
df = df[(z_scores < 3).all(axis=1)]

scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=np.number).columns
numerical_cols = numerical_cols.drop('Survived')
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
