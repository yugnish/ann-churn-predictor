import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping


df=pd.read_csv("data/processed_churn.csv")
X=df.drop("Churn",axis=1)
y=df["Churn"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
print(X_train.shape)
model=Sequential()
model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )

)
model.add(
    Dropout(0.3)
)
model.add(
    Dense(
        32,
        activation="relu"
    )
)
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
early_stop=EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


history=model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
loss,accuracy=model.evaluate(
    X_test,
    y_test
)



print("Test Accuracy:",accuracy)
model.save(
    "models/ann_model.h5"
)