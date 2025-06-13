import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

data = pd.read_csv('finalized_combined.csv')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]

y_train = clean_dataset(data)
data["winning_alliance"].replace({"red": 0, "blue": 1}, inplace=True)

print(data.head())

cols = ["red1_epa_end", "red2_epa_end", "red3_epa_end", "blue1_epa_end", "blue2_epa_end", "blue3_epa_end"]

X = data[cols]
y = data["winning_alliance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logReg = LogisticRegression(random_state=16)

logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
