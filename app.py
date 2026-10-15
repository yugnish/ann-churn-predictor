# from flask import Flask,render_template,request
# from tensorflow.keras.models import load_model
# import pickle
# import numpy as np



# app=Flask(__name__)


# model=load_model("models/ann_model.h5")


# with open(
#     "models/scaler.pkl",
#     "rb"
# ) as file:
#     scaler=pickle.load(file)


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict",methods=["POST"])
# def predict():
#     @app.route("/predict", methods=["POST"])
# def predict():

#     # Get values from HTML form

#     gender = request.form["gender"]
#     SeniorCitizen = request.form["SeniorCitizen"]
#     Partner = request.form["Partner"]
#     Dependents = request.form["Dependents"]

#     tenure = request.form["tenure"]

#     PhoneService = request.form["PhoneService"]
#     MultipleLines = request.form["MultipleLines"]

#     InternetService = request.form["InternetService"]
#     OnlineSecurity = request.form["OnlineSecurity"]
#     OnlineBackup = request.form["OnlineBackup"]

#     DeviceProtection = request.form["DeviceProtection"]
#     TechSupport = request.form["TechSupport"]

#     StreamingTV = request.form["StreamingTV"]
#     StreamingMovies = request.form["StreamingMovies"]

#     Contract = request.form["Contract"]

#     PaperlessBilling = request.form["PaperlessBilling"]

#     PaymentMethod = request.form["PaymentMethod"]

#     MonthlyCharges = request.form["MonthlyCharges"]

#     TotalCharges = request.form["TotalCharges"]



#     # Create input in SAME ORDER as training

#     customer_data = [
#         gender,
#         SeniorCitizen,
#         Partner,
#         Dependents,
#         tenure,
#         PhoneService,
#         MultipleLines,
#         InternetService,
#         OnlineSecurity,
#         OnlineBackup,
#         DeviceProtection,
#         TechSupport,
#         StreamingTV,
#         StreamingMovies,
#         Contract,
#         PaperlessBilling,
#         PaymentMethod,
#         MonthlyCharges,
#         TotalCharges
#     ]


#     # Convert strings to float

#     customer_data = np.array(
#         customer_data,
#         dtype=float
#     )


#     # Reshape for ANN

#     customer_data = customer_data.reshape(
#         1,-1
#     )


#     # Apply scaling

#     customer_data = scaler.transform(
#         customer_data
#     )


#     # ANN prediction

#     prediction = model.predict(
#         customer_data
#     )


#     if prediction[0][0] > 0.5:

#         result = "Customer will churn"

#     else:

#         result = "Customer will stay"



#     return result

# if __name__=="__main__" :
#     app.run(debug=True)
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("models/ann_model.h5")

# ann model doesnot work on values given by user,so scaler scales the values
with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")

# taking the inputs given by user
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Extract features in exact order of training data
    raw_features = [
        request.form["gender"],
        request.form["SeniorCitizen"],
        request.form["Partner"],
        request.form["Dependents"],
        request.form["tenure"],
        request.form["PhoneService"],
        request.form["MultipleLines"],
        request.form["InternetService"],
        request.form["OnlineSecurity"],
        request.form["OnlineBackup"],
        request.form["DeviceProtection"],
        request.form["TechSupport"],
        request.form["StreamingTV"],
        request.form["StreamingMovies"],
        request.form["Contract"],
        request.form["PaperlessBilling"],
        request.form["PaymentMethod"],
        request.form["MonthlyCharges"],
        request.form["TotalCharges"],
    ]

    # 2. Convert to numpy array & float
    # here raw data is in list and values inside it are strings,so we are converting it in array and values as integers and scaler acts only on 2d data 
    # values should be integer inside it
    data = np.array(raw_features, dtype=float).reshape(1, -1)

    # 3. Apply standard scaler
    data_scaled = scaler.transform(data)

    # 4. Make prediction
    prediction = model.predict(data_scaled)
    churn_probability = float(prediction[0][0])

    # 5. Determine prediction result message
    if churn_probability > 0.5:
        result = f"Customer is likely to CHURN (Probability: {churn_probability:.2%})"
    else:
        result = f"Customer is likely to STAY (Probability: {(1 - churn_probability):.2%})"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)