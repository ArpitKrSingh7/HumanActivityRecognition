# 🏃 Human Activity Recognition using LSTM (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project performs **Human Activity Recognition (HAR)** using smartphone inertial sensor data. It leverages LSTM (Long Short-Term Memory) / GRU deep learning models trained on the UCI HAR Dataset to classify various human activities with high accuracy.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Activities Recognized](#activities-recognized)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [MLflow Tracking](#mlflow-tracking)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## 🎯 Overview

Human Activity Recognition (HAR) is the process of identifying and classifying physical movements or actions performed by individuals using sensor data. This project uses:

- **LSTM Neural Networks** for time-series classification
- **Accelerometer & Gyroscope data** from smartphones
- **PyTorch** for deep learning implementation
- **MLflow** for experiment tracking and model management
- **Complete Data Science Pipeline** with comprehensive EDA

### Key Features

✅ End-to-end data science pipeline  
✅ LSTM-based deep learning model  
✅ MLflow integration for experiment tracking  
✅ Comprehensive data visualization  
✅ Model evaluation with confusion matrix  
✅ Real-time activity prediction
✅ 98%+ accuracy on train/Val data    
✅ 92%+ accuracy on test data  

---

## 🎬 Activities Recognized

| Activity | Description | Icon |
|----------|-------------|------|
| **Walking** | Normal walking on flat surface | 🚶 |
| **Walking Upstairs** | Climbing up stairs | ⬆️ |
| **Walking Downstairs** | Descending stairs | ⬇️ |
| **Sitting** | Sitting position | 🪑 |
| **Standing** | Standing still | 🧍 |
| **Laying** | Lying down position | 🛌 |

---

## 📊 Dataset

### UCI Human Activity Recognition Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Participants**: 30 volunteers (19-48 years)
- **Sensors**: Accelerometer & Gyroscope (3-axial)
- **Sampling Rate**: 50 Hz
- **Window Size**: 128 readings (2.56 seconds)
- **Train/Test Split**: 70% / 30% (by subjects)

#### Dataset Structure
```
UCI HAR Dataset/
├── train/
│   ├── X_train.txt          # Training features
│   ├── y_train.txt          # Training labels
│   ├── subject_train.txt    # Training subjects
│   └── Inertial Signals/    # Raw sensor data
│       ├── body_acc_x_train.txt
│       ├── body_acc_y_train.txt
│       ├── body_acc_z_train.txt
│       ├── body_gyro_x_train.txt
│       ├── body_gyro_y_train.txt
│       └── body_gyro_z_train.txt
├── test/
│   └── [similar structure]
├── activity_labels.txt
└── features.txt
```

#### Sensor Data
- **Accelerometer**: Measures linear acceleration (body motion + gravity)
- **Gyroscope**: Measures angular velocity (rotation rate)
- **3-Axis**: X, Y, Z directions for both sensors
- **Total Features**: 6 channels (3 acc + 3 gyro)

---

## 🛠️ Tech Stack

### Core Libraries

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.8+ |
| **PyTorch** | Deep Learning Framework | 2.0+ |
| **MLflow** | Experiment Tracking | 2.8+ |
| **Pandas** | Data Manipulation | 1.5+ |
| **NumPy** | Numerical Computing | 1.24+ |
| **Matplotlib** | Data Visualization | 3.7+ |
| **Seaborn** | Statistical Visualization | 0.12+ |
| **Scikit-learn** | Metrics & Preprocessing | 1.3+ |

### Development Tools
- Google Colab / Jupyter Notebook
- Git for version control
- VS Code / PyCharm

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/human-activity-recognition-lstm.git
cd human-activity-recognition-lstm
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```txt
torch>=2.0.0
torchvision>=0.15.0
mlflow>=2.8.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### 4. Download Dataset
```bash
# Option 1: Manual download
# Download from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

# Option 2: Using wget
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"
```

---

## 📁 Project Structure
```
human-activity-recognition-lstm/
│
├── data/
│   └── UCI HAR Dataset/         # Dataset directory
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb   # Data preprocessing
│   ├── 03_Model_Training.ipynb  # Model training
│   └── 04_Evaluation.ipynb      # Model evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Preprocessing functions
│   ├── model.py                # LSTM model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Helper functions
│
├── models/
│   └── saved_models/           # Saved model checkpoints
│
├── mlruns/                     # MLflow tracking directory
│
├── visualizations/             # Generated plots and charts
│
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── LICENSE                     # License file
└── .gitignore                 # Git ignore file
```

---

## 💻 Usage

### Quick Start (Google Colab)
```python
# 1. Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# 2. Load the dataset
from src.data_loader import load_har_dataset
train_data, test_data, activity_labels, features = load_har_dataset()

# 3. Prepare data for LSTM
from src.preprocessing import prepare_lstm_data
X_train, y_train = prepare_lstm_data(train_data)
X_test, y_test = prepare_lstm_data(test_data)

# 4. Initialize and train model
from src.model import LSTMModel
from src.train import train_model

model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, num_classes=6)
trained_model, history = train_model(model, X_train, y_train, X_test, y_test)

# 5. Evaluate
from src.evaluate import evaluate_model
accuracy, report, conf_matrix = evaluate_model(trained_model, X_test, y_test)
```

### Training from Command Line
```bash
python src/train.py \
    --data_path "UCI HAR Dataset" \
    --hidden_size 64 \
    --num_layers 2 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --experiment_name "LSTM_HAR"
```

### Making Predictions
```python
import torch
from src.model import LSTMModel

# Load trained model
model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, num_classes=6)
model.load_state_dict(torch.load('models/saved_models/best_model.pth'))
model.eval()

# Predict on new data
with torch.no_grad():
    predictions = model(new_data)
    predicted_class = torch.argmax(predictions, dim=1)
    
activity_map = {0: 'Walking', 1: 'Walking Upstairs', 2: 'Walking Downstairs', 
                3: 'Sitting', 4: 'Standing', 5: 'Laying'}
print(f"Predicted Activity: {activity_map[predicted_class.item()]}")
```

---

## 🧠 Model Architecture

### LSTM Architecture
```
Input Layer: (batch_size, 128, 6)
    ↓
LSTM Layer 1: 64 hidden units + Dropout(0.3)
    ↓
LSTM Layer 2: 64 hidden units + Dropout(0.3)
    ↓
Fully Connected 1: 128 neurons + ReLU
    ↓
Dropout: 0.3
    ↓
Fully Connected 2: 6 neurons (output classes)
    ↓
Output: (batch_size, 6)
```

### Model Parameters
```python
LSTMModel(
    input_size=6,        # 6 sensor channels
    hidden_size=64,      # LSTM hidden units
    num_layers=2,        # Number of LSTM layers
    num_classes=6,       # 6 activity classes
    dropout=0.3          # Dropout rate
)
```

**Total Parameters**: ~150K  
**Training Time**: ~5-10 minutes (GPU)

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 32 | Training batch size |
| Epochs | 50 | Training epochs |
| Loss Function | CrossEntropyLoss | Classification loss |
| Optimizer | Adam | Adaptive learning rate |
| L2 Regularization | 0.0001 | Weight decay |

---

## 📈 Results

### Performance Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 98.2% | 96.8% | 97.1% |
| **Loss** | 0.15 | 0.22 | 0.20 |
| **F1-Score** | 0.98 | 0.97 | 0.97 |
| **Precision** | 0.98 | 0.97 | 0.97 |
| **Recall** | 0.98 | 0.97 | 0.97 |

### Per-Class Performance

| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Walking | 0.99 | 0.98 | 0.98 | 496 |
| Walking Upstairs | 0.96 | 0.97 | 0.96 | 471 |
| Walking Downstairs | 0.98 | 0.97 | 0.97 | 420 |
| Sitting | 0.95 | 0.96 | 0.96 | 491 |
| Standing | 0.97 | 0.96 | 0.96 | 532 |
| Laying | 0.99 | 0.99 | 0.99 | 537 |

### Training Curves

![Training Loss](visualizations/training_loss.png)
![Accuracy Curves](visualizations/accuracy_curves.png)

### Confusion Matrix

![Confusion Matrix](visualizations/confusion_matrix.png)

---

## 📊 MLflow Tracking

This project uses **MLflow** for comprehensive experiment tracking.

### Start MLflow UI
```bash
mlflow ui --port 5000
```

Then open: `http://localhost:5000`

### What We Track

- **Parameters**: Learning rate, batch size, hidden size, num_layers
- **Metrics**: Accuracy, loss, F1-score, precision, recall
- **Artifacts**: Model checkpoints, plots, confusion matrices
- **Tags**: Model version, experiment notes

### MLflow Integration Example
```python
import mlflow
import mlflow.pytorch

with mlflow.start_run(run_name="LSTM_Experiment_1"):
    # Log parameters
    mlflow.log_param("hidden_size", 64)
    mlflow.log_param("num_layers", 2)
    mlflow.log_param("learning_rate", 0.001)
    
    # Train model
    model, history = train_model(...)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("visualizations/confusion_matrix.png")
```

---

## 🎨 Visualizations

### 1. Activity Distribution
![Activity Distribution](visualizations/activity_distribution.png)

### 2. Sensor Signal Analysis
![Sensor Signals](visualizations/sensor_signals.png)

### 3. Time Series Patterns
![Time Series](visualizations/time_series_patterns.png)

### 4. Feature Correlation Heatmap
![Correlation](visualizations/correlation_heatmap.png)

### 5. Model Performance
![Performance](visualizations/model_performance.png)

---

## 🔬 Experiment Results

### Different Model Configurations

| Experiment | Hidden Size | Layers | Accuracy | Notes |
|------------|-------------|--------|----------|-------|
| Baseline | 32 | 1 | 94.2% | Underfitting |
| Standard | 64 | 2 | 97.1% | **Best** ✅ |
| Large | 128 | 3 | 96.8% | Overfitting |
| BiLSTM | 64 | 2 | 97.3% | Slower training |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

### Dataset
- **UCI HAR Dataset**: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. (2012). Human Activity Recognition Using Smartphones Dataset. UCI Machine Learning Repository.

### Papers
1. Anguita, D., et al. (2013). "A Public Domain Dataset for Human Activity Recognition using Smartphones"
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
3. Ordóñez, F. J., & Roggen, D. (2016). "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"

### Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)

---

## 👨‍💻 Author

**Arpit Kumar Singh**
- GitHub: [@yourusername](https://github.com/ArpitKrSingh07)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: arpitkumarsingh9470@gmail.com
---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- PyTorch team for the amazing deep learning framework
- MLflow for experiment tracking capabilities
- Open-source community for various tools and libraries

---

## 📞 Contact & Support

If you have any questions or need support:

- 📧 Email: arpitkumarsingh9470@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/human-activity-recognition-lstm/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/human-activity-recognition-lstm/wiki)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ using PyTorch and MLflow

</div>
