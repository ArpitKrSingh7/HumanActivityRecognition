import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import joblib
from sklearn.preprocessing import StandardScaler


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]     
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def load_components(model_path="models/best_model/best_model.pth"):
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    input_size = 9   # 9 features per timestep
    hidden_size = 128
    num_layers = 2
    num_classes = len(label_encoder.classes_)

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler, label_encoder


def predict(csv_path):
    model, scaler, label_encoder = load_components()

    data = pd.read_csv(csv_path, header=None)
    data = data.values.astype(float)

    # ✅ Expect flattened input (1×1152)
    if data.shape[1] != 1152:
        raise ValueError(
            f"❌ Input has {data.shape[1]} features, but model expects 1152 "
            "(128 timesteps × 9 features)"
        )

    # ✅ Scale
    data_scaled = scaler.transform(data)

    # ✅ reshape back to (batch, seq_len, features)
    timesteps = 128
    features = 9
    data_reshaped = data_scaled.reshape(1, timesteps, features)

    input_tensor = torch.tensor(data_reshaped, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    activity = label_encoder.inverse_transform([pred])[0]
    confidence = float(probs[0][pred] * 100)

    print("\n✅ Prediction Done")
    print(f"🏃 Activity: **{activity}**")
    print(f"🎯 Confidence: {confidence:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAR Prediction Script")
    parser.add_argument("--file", required=True, help="Path to test CSV with sensor readings")
    args = parser.parse_args()
    predict(args.file)

