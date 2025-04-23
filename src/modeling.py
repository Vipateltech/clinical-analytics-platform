from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def segment_patients(df, features, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = model.fit_predict(df[features])
    return df

def predict_readmission(df, features, target='Readmission'):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model
