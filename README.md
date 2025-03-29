# **OpenPose-Fall-Detection**

Implementing fall detection using the OpenPose algorithm. Integrated with a voice alert system that alerts when a person falls.

## 🚀 Features

- Real-time fall detection using the OpenPose algorithm.
- Voice alert system for immediate notifications.
- Ideal for environments where quick detection and response are critical.

## 🕠️ Installation

### 1 Clone the Repository

```sh
git clone https://github.com/KrishBhimani/Openpose-Fall-Detection.git
```

### 2 Create a Virtual Environment

#### For Windows:
```sh
conda create -p venv python=3.7 -y
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

## 📌 Usage

1. Clone the repository and set up a virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the project with `python main.py`.
4. Monitor the console output for fall detection alerts.

## 🔧 Technologies Used

- **Python**: Programming language for script development.
- **OpenPose**: Real-time multi-person pose estimation library.
- **VoiceAlertSystem**: Custom script to handle voice alerts.

## 🚀 Challenges & Solutions

- **Challenge**: Integrating OpenPose with a custom fall detection algorithm.
  - **Solution**: Customized model training with OpenPose data for accurate pose estimation.
- **Challenge**: Ensuring low latency in voice alerts.
  - **Solution**: Optimized the audio processing pipeline to minimize delay.

## 🤝 Contributing

Contributions are welcome! Feel free to submit **issues** or **pull requests** to improve this project.