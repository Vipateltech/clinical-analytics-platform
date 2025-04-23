import pandas as pd
from sklearn.preprocessing import StandardScaler

p = "C://Users//thinkpad//Desktop//project//clinical_financial_data.csv"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df = df[df['BMI'] > 0]
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

d = load_data(p)
cd = clean_data(d)
features = [
    "Age",
    "BMI",
    "BloodPressure",
    "Glucose",
    "Insulin",
    "BillingAmount"
]
print(scale_features(cd,features))