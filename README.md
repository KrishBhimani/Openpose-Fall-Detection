# **OpenPose Fall Detection**

Implementing fall detection using the Openpose algorithm. Integrated voice alert system which alerts when a person falls.

## 🚀 Features

- Real-time fall detection using OpenPose.
- Voice alerts activated upon detecting a fall.
- Customizable alert messages and detection sensitivity.

## 🕠️ Installation

### 1 Clone the Repository

```sh
git clone https://github.com/KrishBhimani/Openpose-Fall-Detection
```

### 2 Create a Virtual Environment

#### For Windows:
```sh
conda create -p venv python==3.11 -y
conda activate venv/
```

#### For macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3 Install Dependencies

```sh
pip install -r requirements.txt
```

### 4 Run the Application

```sh
python main.py
```

## 🔧 Technologies Used

- **Python**: Primary programming language.
- **OpenPose**: For real-time keypoint detection.
- **PyTorch**: For machine learning and deep learning models.
- **TensorFlow**: For additional deep learning operations and models.

## 🚀 Challenges & Solutions

- **Challenge**: Dealing with noisy data and real-time processing.
  - **Solution**: Leveraging advanced filtering techniques and optimizing OpenPose.
- **Challenge**: Integrating voice alerts.
  - **Solution**: Utilizing text-to-speech libraries for generating real-time alerts.

---
This README file has been generated based on the available information and standard practices for real-time fall detection applications using OpenPose. Ensure you run the repository within the specified virtual environment and follow the installation steps carefully.