import numpy as np
import pandas as pd

# Paths to inertial test signals
path = "data/UCIHARDataset/test/Inertial Signals/"

files = [
    "body_acc_x_test.txt", "body_acc_y_test.txt", "body_acc_z_test.txt",
    "body_gyro_x_test.txt", "body_gyro_y_test.txt", "body_gyro_z_test.txt",
    "total_acc_x_test.txt", "total_acc_y_test.txt", "total_acc_z_test.txt"
]

data = []

# Load each signal file into array
for f in files:
    arr = np.loadtxt(path + f)
    data.append(arr)

# Stack into shape (samples, 9, 128)
data = np.stack(data, axis=2)

# Now flatten the first sample → shape (1, 128*9)
sample = data[0].reshape(1, -1)

df = pd.DataFrame(sample)
df.to_csv("sample_from_dataset.csv", index=False, header=False)

print("✅ Exported sample_from_dataset.csv (1×1152) for prediction.")

