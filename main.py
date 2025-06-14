import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('finalized_combined.csv')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]

y_train = clean_dataset(data)
data["winning_alliance"].replace({"red": 0, "blue": 1}, inplace=True)

X = data.drop(columns=["winning_alliance", "match_number", "year", "match_key", "event_key", "match_type", "set_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"])
y = data["winning_alliance"]

X["red_epa_sum"] = X["red1_epa_end"] + X["red2_epa_end"] + X["red3_epa_end"]
X["blue_epa_sum"] = X["blue1_epa_end"] + X["blue2_epa_end"] + X["blue3_epa_end"]
X["epa_ratio"] = X["red_epa_sum"] / X["blue_epa_sum"]

variances = np.var(X, axis=0)

vt = VarianceThreshold(threshold=0.3)

X_reduced = vt.fit_transform(X)

selected_columns = X.columns[vt.get_support()]
X = X[selected_columns]

clf = RandomForestClassifier()

clf.fit(X, y)
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
features_to_drop = feature_importances.sort_values(ascending=False).index.tolist()[-32:-1]
X.drop(features_to_drop, axis=1, inplace=True)

samples, features = X.shape
print(f"Samples: {samples}, Features: {features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(features)

learning_rate = 0.1
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

num_epochs = 1000

for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    scheduler.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc:.4f}')