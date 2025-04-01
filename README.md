# **Openpose-Fall-Detection**

Implementing fall detection using the OpenPose algorithm. Integrated with a voice alert system which alerts when a person falls, ensuring timely assistance.

## 🚀 Features

- **Fall Detection:** Real-time monitoring using the OpenPose algorithm to detect a fall.
- **Voice Alert:** Automatic voice alert when a fall is detected.
- **User-Friendly:** Easy setup and usage.

## 🕠️ Installation

### 1 Clone the Repository

```sh
git clone https://github.com/KrishBhimani/Openpose-Fall-Detection.git
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

- **OpenPose:** Human pose estimation library.
- **Python:** General-purpose programming language for implementation.
- **NumPy:** Library for numerical computations in Python.
- **PyTorch:** Deep learning framework, if used for AI model integration.
  
## 🚀 Challenges & Solutions

- **Challenge 1:** Integrating real-time video input for fall detection.
  - **Solution:** Utilized OpenCV for real-time video processing and frame-by-frame analysis.
  
- **Challenge 2:** Ensuring prompt and accurate sound alerts.
  - **Solution:** Implemented a voice alert system using Python's `pyttsx3` library which converts text to speech.

---

## Contribution Guidelines

Contributions are always welcome. If you have any improvements or enhancements, please open an issue or directly send a pull request. Be sure to follow the existing style and add proper documentation for any new features.
  
Feel free to explore the source code and make meaningful contributions to the project.

Enjoy!