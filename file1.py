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

#insonsistent
print(df["gender"].unique())
print(df["country"].unique())
print(df["device_type"].unique())
print(df["preferred_category"].unique())

print(df.isnull().sum())
print("Duplicate Values:",df.duplicated().sum())

#conversion
df["signup_date"]=pd.to_datetime(df["signup_date"])
df["last_purchase_date"]=pd.to_datetime(df["last_purchase_date"])

print(df.head())
print(df.info())
print(df.describe())

#objectives

#1. Analyse dist of customers who churn and don't churn
sns.countplot(x="churned",data=df)           #bar chart show how many in churn and not churn
plt.title("Customer Churn Distribution")
plt.show()

#2. Checking customer spending is related to churn or not
sns.boxplot(x="churned",y="total_spent",data=df)
plt.title("Total Spending and Customer Churn")
plt.show()

#3. Checking customer recency is related to churn
sns.boxplot(x="churned",y="recency_days",data=df)
plt.title("Recency Days and Customer Churn")
plt.show()

#4.analyze premium membership is related to customer churn
sns.countplot(x="is_premium_member",hue="churned",data=df)
plt.title("Premium Membership and Customer Churn")  #compare btw premium and non premium customers
plt.show()

#5. checkinf preferred product related to customer churn

sns.countplot(x="preferred_category",hue="churned",data=df)
plt.title("Preferred Category and Customer Churn")
plt.show()

