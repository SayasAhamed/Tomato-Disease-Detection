# 🍅 Tomato Disease Detection System (Multi-Model AI Web App)

A **Flask-based AI web application** that detects tomato leaf diseases using deep learning models.
Users can upload an image of a tomato leaf and receive:

* 🧠 Predicted disease class
* 📊 Confidence score
* 📖 Disease information

---

## 📌 Overview

This project uses **computer vision + deep learning** to identify common tomato plant diseases.
It supports multiple models including:

* MobileNet (lightweight & fast)
* ResNet50 (high accuracy)

---

## 🚀 Features

* 📸 Upload tomato leaf images
* 🤖 AI-powered disease prediction
* 📊 Confidence score output
* 📖 Disease descriptions
* 🔄 Multi-model support
* 🌐 Simple web interface (Flask)

---

## 🛠️ Tech Stack

* 🐍 Python
* 🌐 Flask
* 🤖 TensorFlow / Keras
* 🧠 Deep Learning (CNN)
* 🖼️ OpenCV
* 📊 NumPy

---

## 📁 Project Structure

```bash id="x7r2n1"
Tomato-disease-detection/
│
├── app.py
├── train_mobilenet.py
├── train_resnet50.py
├── clean_dataset.py
├── README.md
│
├── templates/
├── static/
├── uploads/
│
├── utils/
│   └── disease_info.py
│
└── model/   (create this manually)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash id="2b6k6m"
git clone https://github.com/yourusername/Tomato-disease-detection.git
cd Tomato-disease-detection
```

---

### 2️⃣ Install Dependencies

```bash id="b6szl3"
pip install -r requirements.txt
```

---

## 📦 Dataset & Model Setup (IMPORTANT)

⚠️ The dataset and trained models are NOT included in this repository.

### 👉 Follow these steps:

1. **Create a folder named:**

```bash id="zq1m3g"
model/
```

2. Download the dataset from Google Drive
   (you will add your link later)

3. Place the downloaded file:

```bash id="g3k0hx"
Data.zip
```

inside the `model/` folder

4. Extract it:

```bash id="v7zhl0"
model/Data.zip → extract here
```

👉 After extraction, structure should look like:

```bash id="hhp1x6"
model/
│
├── train/
└── test/
```

---

## 🧠 Training Models

You can train your own models using:

### 🔹 MobileNet

```bash id="6x1y4l"
python train_mobilenet.py
```

### 🔹 ResNet50

```bash id="psmnjx"
python train_resnet50.py
```

👉 This will generate `.h5` model files inside the `model/` folder

---

## ▶️ Run the Application

```bash id="6rqf7c"
python app.py
```

Open in browser:

```bash id="t9p3tf"
http://127.0.0.1:5000/
```

---

## 🧪 How It Works

1. User uploads a tomato leaf image
2. Image is preprocessed using OpenCV
3. Selected model (MobileNet / ResNet50) predicts disease
4. Result + confidence displayed on UI

---

## ⚠️ Notes

* ❌ Dataset is not included (too large for GitHub)
* ❌ Model files are not included
* ✅ Upload your own dataset and train locally

---

## 🚧 Future Improvements

* 🔔 Real-time disease alerts
* 📱 Mobile-friendly UI
* 🌐 Deploy to cloud (Render / HuggingFace)
* 🧠 Add more crop disease models
* 📊 Accuracy comparison dashboard

---

## 👨‍💻 Author

**M.M. Sayas Ahamed**
🎓 BICT Undergraduate
💻 AI & Full Stack Developer
🎥 Tech Content Creator

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🛠️ Contribute

---

## 📄 License

This project is for educational purposes only.
