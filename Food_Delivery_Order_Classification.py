import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import kagglehub
from sklearn.neighbors import KNeighborsClassifier     

path = kagglehub.dataset_download("sujalsuthar/food-delivery-order-history-data")

print("Path to dataset files:",path + '\\order_history_kaggle_data.csv')

#Load the Data
df = pd.read_csv(path + '\\order_history_kaggle_data.csv')


#Select 1000 sample records
df = df.sample(n=5000, random_state=42)


#Select 10 columns
df=df[["Distance","Bill subtotal","Packaging charges","Gold discount",
      "Restaurant discount (Promo)","Total","Rating","KPT duration (minutes)",
      "Rider wait time (minutes)","Order Status"]]
print(df.head())

df.columns=(
    df.columns
    .str.lower()
    .str.replace(" ","_")
    .str.replace("(","")
    .str.replace(")","")
)


print(df["order_status"].value_counts()) #imbalanced data


#class distribution before balancing
plt.figure()
df['order_status'].value_counts().plot(kind='bar')
plt.title("Class Distribution (Before Balancing)")
plt.xlabel("Order Status")
plt.ylabel("Count")
plt.show()


#Convert target variable Delivered==1 or Rejected or Returned==0
df["order_status"]=df["order_status"].apply(
    lambda x:1 if x=="Delivered" else 0
)
print(df["order_status"].value_counts())

#Handling imbalanced data
df_majority = df[df["order_status"] == 1]
df_minority = df[df["order_status"] == 0]

df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_balanced["order_status"].value_counts())

plt.figure()
df_balanced['order_status'].value_counts().plot(kind='bar')
plt.title("Class Distribution (After Balancing)")
plt.show()

#Convert distance feature to numeric eg:5km to 5
print(df_balanced["distance"].unique())

df_balanced["distance"] = df_balanced["distance"].str.replace("km", "")
df_balanced["distance"] = df_balanced["distance"].replace("<1", "0.5")
df_balanced["distance"] = df_balanced["distance"].astype(float)

print(df_balanced["distance"].unique()) 

#Check missing values
print(df_balanced.isnull().sum())

#Handling missing values
#Drop High missing Column
df_balanced=df_balanced.drop(columns=["rating"])

#Filling missing values using median
df_balanced["distance"].fillna(df_balanced["distance"].median(),inplace=True)
df_balanced["kpt_duration_minutes"].fillna(df_balanced["kpt_duration_minutes"].median(),inplace=True)
df_balanced["rider_wait_time_minutes"].fillna(df_balanced["rider_wait_time_minutes"].median(),inplace=True)


cols = ['distance', 'bill_subtotal', 'total']

for col in cols:
    plt.figure()
    plt.hist(df_balanced[col], bins=30)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


    #Correlation heatmap
plt.figure()
sns.heatmap(df_balanced.corr(numeric_only=True), annot=True)
plt.title("Feature Correlation")
plt.show()


#Train Test split without sklearn

from sklearn.model_selection import train_test_split

X = df_balanced.drop(columns=["order_status"]).values
y = df_balanced["order_status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = knn_model.score(X_test, y_test)
print("KNN Accuracy:", accuracy)
from sklearn.metrics import classification_report, confusion_matrix 
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  
