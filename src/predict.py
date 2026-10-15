import numpy as np
import pickle


from tensorflow.keras.models import load_model
model=load_model("models/ann_model.h5")


with open(
    "models/scaler.pkl",
    "rb"
)as file:
    scaler=pickle.load(file)

def predict_churn(customer_data):
    customer_data=np.array(
        customer_data
    ).reshape(1,-1)
    customer_data=scaler.transform(
        customer_data
    )
    prediction=model.predict(
        customer_data
    )

