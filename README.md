# 🩺 Diabetes Prediction using Supervised Learning

## 🔍 Project Overview
This project applies **supervised machine learning algorithms** to predict whether a patient is diabetic based on diagnostic medical attributes. Using the **Pima Indians Diabetes Dataset**, we trained and evaluated multiple classification models to identify high-risk individuals and assist in early detection.

---

## 📊 Dataset
- **Source:** [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** Outcome (0: Non-Diabetic, 1: Diabetic)

---

## 🚀 Features Implemented
- ✅ Data cleaning & exploration  
- ✅ Feature scaling with `StandardScaler`  
- ✅ Model training: Logistic Regression, Random Forest, and SVM  
- ✅ Model evaluation using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix & Classification Report
  - ROC Curve & AUC Score  
- ✅ Single-patient prediction with real data simulation  
- ✅ Clean, modular, and well-commented code

---

## 🧠 Algorithms Used
| Model                  | Description                                      |
|------------------------|--------------------------------------------------|
| **Logistic Regression** | Interpretable baseline classifier               |
| **Random Forest**       | Ensemble method for robust predictions          |
| **Support Vector Machine (SVM)** | Effective for small-to-medium datasets with scaling |

---

## 📈 Performance
The **Random Forest** classifier showed the best performance with:
- **Accuracy:** ~85%  
- **ROC AUC Score:** High discriminative power  
- **Balanced precision and recall**, ideal for medical diagnosis

---

## 🔮 Sample Prediction
```python
sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
```
---
## 🛠️ Tech Stack
- Python (`NumPy`, `Pandas`, `Scikit-Learn`)
- `Matplotlib` & `Seaborn` for visualizations
- Jupyter Notebook / Google Colab

---
###💡 Future Improvements
- Hyperparameter tuning using GridSearchCV
- Model deployment with Streamlit or Flask
- Cross-validation and imputation for missing values
- Advanced models like XGBoost or LightGBM

  
---


