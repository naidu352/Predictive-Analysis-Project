import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score

df=pd.read_csv("ecommerces.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

print("Missing Values:")
print(df.isnull().sum())
print("duplicate values:",df.duplicated().sum())

#preprocess

df["age"]=df["age"].fillna(df["age"].median())
df["total_spent"]=df["total_spent"].fillna(df["total_spent"].median())
df["gender"]=df["gender"].fillna(df["gender"].mode()[0])
df["country"]=df["country"].fillna(df["country"].mode()[0])
df["device_type"]=df["device_type"].fillna(df["device_type"].mode()[0])
df["preferred_category"]=df["preferred_category"].fillna(df["preferred_category"].mode()[0])

df=df.drop_duplicates()
print("Missing Values:")
print(df.isnull().sum())
print("Duplicate Values:",df.duplicated().sum())

