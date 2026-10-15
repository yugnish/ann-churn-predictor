import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model


from sklearn.model_selection import train_test_split

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)

df=pd.read_csv("data/processed_churn.csv")

X=df.drop("Churn",axis=1)

y=df["Churn"]


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=load_model(
    "models/ann_model.h5"
)

y_probability=model.predict(X_test)

y_pred=(y_probability >0.5).astype(int)


accuracy=accuracy_score(
    y_test,
    y_pred
)

print("accuracy: ",accuracy)

print("\nclassification report: ")
print(
    classification_report(
        y_test,
        y_pred
    )
)