
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('sample_titanic_dataset.csv')
print("Dataset Loaded Successfully!\n")
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
print("\nSummary Statistics:")
print(df.describe())
sns.countplot(x='survived', data=df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Gender')
plt.show()
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print("\nSurvival Rate by Gender:")
print(df.groupby('sex')['survived'].mean())
print("\nSurvival Rate by Passenger Class:")
print(df.groupby('pclass')['survived'].mean())
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

print("\nSurvival Rate by Age Group:")
print(df.groupby('age_group')['survived'].mean())

sns.barplot(x='age_group', y='survived', data=df)
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.show()
df.to_csv('cleaned_sample_titanic_dataset.csv', index=False)
print("Cleaned dataset exported as 'cleaned_sample_titanic_dataset.csv'")

