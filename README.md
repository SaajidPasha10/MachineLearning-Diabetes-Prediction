# Diabetes Prediction App

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A web application built with **Streamlit** that predicts the likelihood of diabetes based on user inputs. Users can select different machine learning models (Logistic Regression, Random Forest, XGBoost) and see both **predictions** and **probabilities**.

---

## 🌟 Features

- Select **Machine Learning Model** from a dropdown.
- Input patient data:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Get **prediction**: Positive / Negative
- View **probability** of diabetes
- Visual **progress bar** indicating risk level
- Models trained with **Pima Indians Diabetes Dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Pre-trained models: Logistic Regression, Random Forest, XGBoost


## 🚀 Live Demo

Access the app online: [Diabetes Prediction Streamlit App](https://saajidpasha10-machinelearning-diabetes-prediction-app-0vezck.streamlit.app/)  
*(Replace with your deployed Streamlit link)*

---

## 💻 Installation

1. **Clone the repository:**

```
git clone https://github.com/yourusername/diabetes-prediction-app.git
cd diabetes-prediction-app
```
Create a virtual environment:
```
python -m venv venv
```
# Linux/macOS
```
source venv/bin/activate
```
# Windows
```
venv\Scripts\activate

```
### Install dependencies:
```
pip install -r requirements.txt
```

Run the Streamlit app:
```
streamlit run app.py
```

Open in browser: http://localhost:8501
Select a model, fill in patient features, and click Predict Diabetes.

📂 Folder Structure
```
diabetes-prediction-app/
│
├── models/                 # Pre-trained ML models (.pkl files)
│   ├── LogisticRegression.pkl
│   ├── RandomForestClassifier.pkl
│   └── XGBClassifier.pkl
│
├── assets/                 # Screenshots, GIFs, images
│   └── demo.gif
├── app.py                  # Streamlit app
├── diabetes.csv            # Dataset (optional)
├── requirements.txt        # Python dependencies
└── README.md
```

### 📊 Model Details
```

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.78	0.74	0.68	0.71	0.82
Random Forest	0.81	0.76	0.73	0.74	0.85
XGBoost	0.83	0.78	0.75	0.76	0.87
```

Metrics calculated using 5-fold cross-validation.

## ⚙️ Contributing

Contributions are welcome!

Fork the repository

Create your feature branch (git checkout -b feature/new-feature)

Commit changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/new-feature)

Open a Pull Request

📜 License

This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements

Kaggle Pima Indians Diabetes Dataset

Streamlit Documentation

Scikit-learn, XGBoost, Seaborn & Matplotlib




