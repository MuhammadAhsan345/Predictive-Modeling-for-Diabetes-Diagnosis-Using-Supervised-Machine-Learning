# 🩺 Diabetes Prediction Using Machine Learning

This project applies supervised machine learning techniques to predict the likelihood of diabetes in female patients based on clinical data. The dataset used is the Pima Indians Diabetes dataset, obtained from Kaggle. The goal is to aid early diagnosis and help healthcare professionals identify high-risk individuals efficiently.

## 📂 Dataset

- **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768 patients
- **Features**: 8 numerical features including Glucose, Insulin, BMI, Blood Pressure, Age, etc.
- **Target Variable**: `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

## ✅ Problem Statement

To build a classification model that can accurately predict whether a patient has diabetes based on medical measurements.

---

## 🛠️ Technologies & Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

---

## 📊 Model Used

- **Random Forest Classifier**  
A powerful ensemble method that performs well with structured tabular data.

---

## 🔍 Model Performance

| Metric              | Value        |
|---------------------|--------------|
| Accuracy            | 73.6%        |
| Precision (Diabetic)| 66%          |
| Recall (Diabetic)   | 52%          |
| F1-Score (Diabetic) | 0.58         |
| ROC AUC Score       | 0.82         |

The model demonstrates high precision for positive diabetes cases, although recall indicates some diabetic cases were missed. Overall, the ROC AUC shows strong discriminative capability.

---

## 📈 Feature Importance

Top contributing features to diabetes prediction (from Random Forest importance analysis):

- Glucose
- BMI
- Age
- Insulin
- Diabetes Pedigree Function

---

## 💡 Key Insights

- Random Forest is effective for structured healthcare data but can be improved for better recall.
- **Recall is prioritized** due to the importance of detecting actual diabetic patients.
- Combining this model with **unsupervised learning** (e.g., anomaly detection, clustering) could help identify new patient subgroups or borderline cases.

---

## 🧪 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-ml.git
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook and run the cells in order:
   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```

   Or use Google Colab for convenience.

---

## 📄 License

This project is open-source and free to use.
