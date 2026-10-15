import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

df=pd.read_csv("data/churn.csv")
print("Original Data:")
print(df.head())

df.drop("customerID",axis=1,inplace=True)


# pd.to_numeric changes the Total charges from string to int
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],errors="coerce"
)
# it drops the  null values rows from the table
df.dropna(inplace=True)

df["gender"]=df["gender"].str.strip()
df["gender"]=df["gender"].str.lower()
encoder=LabelEncoder()
for column in df.columns:
        # encoder.fit_transform changes changes the string values to integers 
    
        df[column]=encoder.fit_transform(df[column])

print("\n After Encoding: ")
print(df.head())



X= df.drop("Churn", axis=1)
y=df["Churn"]
print(df.dtypes)

scaler=StandardScaler()

# it transforms encoded values to the such that their mean becomes 0 and standard deviation to one
X_scaled=scaler.fit_transform(X)
processed_data=pd.DataFrame(
   X_scaled,
   columns=X.columns 
)


processed_data["Churn"]=y.values


processed_data.to_csv(
    "data/processed_churn.csv",
    index=False
)

with open("models/scaler.pkl","wb") as file:
    pickle.dump(scaler,file)
print ("\nPreprocessing Completed Successfully:")