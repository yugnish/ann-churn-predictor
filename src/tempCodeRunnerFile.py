df.drop("customerID",axis=1,inplace=True)

# df["TotalCharges"] = pd.to_numeric(
#     df["TotalCharges"],errors="coerce"
# )
# df.dropna(inplace=True)