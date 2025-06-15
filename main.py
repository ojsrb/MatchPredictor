import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data = pd.read_csv('finalized_combined.csv')

data.dropna(inplace=True)

data["draw"] = data["winning_alliance"] == ""

data["red_win"] = data["winning_alliance"] == "red"
data["blue_win"] = data["winning_alliance"] == "blue"

data.replace({True: 1, False: 0})

X = data.drop(columns=["winning_alliance", "match_number", "year", "match_key", "event_key", "match_type", "set_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"])
y = data[["red_win", "blue_win", "draw"]]

X["red_epa_sum"] = X["red1_epa_end"] + X["red2_epa_end"] + X["red3_epa_end"]
X["blue_epa_sum"] = X["blue1_epa_end"] + X["blue2_epa_end"] + X["blue3_epa_end"]
X["epa_ratio"] = X["red_epa_sum"] / X["blue_epa_sum"]

variances = np.var(X, axis=0)

vt = VarianceThreshold(threshold=0.3)

X_reduced = vt.fit_transform(X)

selected_columns = X.columns[vt.get_support()]
X = X[selected_columns]

bad_features = ['blue3_full_count', 'blue1_state_epa_rank', 'red3_full_count', 'red2_full_count', 'red1_count', 'blue1_count', 'blue2_count', 'red2_count', 'red3_losses', 'red3_full_losses', 'blue3_losses', 'blue2_losses', 'blue1_losses', 'red2_losses', 'red1_losses', 'red2_full_losses', 'blue3_full_losses', 'red1_full_losses', 'blue1_full_losses', 'blue2_full_losses', 'red3_full_ties', 'blue3_full_ties', 'red1_full_ties', 'blue2_full_ties', 'blue2_ties', 'blue3_ties', 'red2_full_ties', 'blue1_full_ties', 'blue1_ties', 'red2_ties', 'red3_ties']
X.drop(bad_features, axis=1, inplace=True)

# Handle feature engineering after scaling
sc = StandardScaler()
X.iloc[:, :] = sc.fit_transform(X)

# Redundant Feature Removal Using Correlation
correlation_matrix = X.corr().abs()
upper_tri = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X = X.drop(columns=to_drop)

samples, features = X.shape
print(f"Samples: {samples}, Features: {features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

y_train = y_train.view(y_train.shape[0], 3)
y_test = y_test.view(y_test.shape[0], 3)

class NeuralNetwork(nn.Module):

    def __init__(self, n_input_features, h_features=60, h_features2=30):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input_features, h_features),
            nn.BatchNorm1d(h_features),
            nn.ReLU(),
            nn.Linear(h_features, h_features2),
            nn.ReLU(),
            nn.Linear(h_features2, h_features2),
            nn.ReLU(),
            nn.Linear(h_features2, 3),  # Note: No activation here, since BCEWithLogitsLoss will handle it
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def train():
    model = NeuralNetwork(n_input_features=features, h_features=60, h_features2=30).to(device)

    learning_rate = 0.01
    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

    num_epochs = 200

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
        y_predicted_cls = torch.sigmoid(y_predicted).round()  # Apply sigmoid for probabilities

        # Calculate accuracy correctly for multi-label classification
        # Count a prediction as correct only if all 3 labels match exactly
        correct = (y_predicted_cls == y_test).all(dim=1).sum().item()
        acc = correct / float(y_test.shape[0])
        print(f'Accuracy: {acc * 100:.4f}%')

train()