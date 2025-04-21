# 🧬 Breast Cancer Classification with Deep Learning and Traditional ML Models

This project aims to classify breast cancer cases as **malignant** or **benign** using a structured dataset. The solution compares three different approaches:  
- A **Deep Neural Network**
- A **Logistic Regression** model
- A **Random Forest** classifier

The goal is to evaluate and benchmark the predictive performance of each model using accuracy, ROC AUC, and feature importance.

---

## 📂 Project Structure

```
├── breast_cancer.ipynb                     # Jupyter Notebook with the full analysis and modeling
├── breast_cancer.csv                       # Dataset (preprocessed)
├── deep_learning_model_training_history.png  # DNN training curves
├── logistic_regression_feature_importance.png
├── model_comparison.png                    # ROC AUC comparison
├── random_forest_feature_importance.png
```

---

## 📈 Results

### ✅ Model Performance (ROC AUC)

| Model              | ROC AUC Score |
|-------------------|---------------|
| Logistic Regression | 0.9918        |
| Random Forest       | 0.9864        |
| **Deep Learning**   | **0.9932** ✅ |

> 🔥 **Best Performance:** Deep Learning Model (DNN) with ROC AUC of **0.9932**

<p align="center">
  <img src="model_comparison.png" width="500">
</p>

---

## 📊 Visual Results

### Deep Learning Model: Accuracy & Loss
<p align="center">
  <img src="deep_learning_model_training_history.png" width="700">
</p>

---

### 🔍 Feature Importance

#### Logistic Regression
<p align="center">
  <img src="logistic_regression_feature_importance.png" width="600">
</p>

#### Random Forest
<p align="center">
  <img src="random_forest_feature_importance.png" width="600">
</p>

---

## 📌 Dataset

- Source: `breast_cancer.csv` (pre-cleaned and numeric)
- Features include: Clump Thickness, Bare Nuclei, Uniformity of Cell Shape, Mitoses, etc.
- Target variable: `class` → Malignant (1) / Benign (0)

---

## 🧠 Technologies & Libraries

- **Python 3**
- **TensorFlow / Keras** – DNN implementation
- **Scikit-learn** – ML models and evaluation
- **Matplotlib / Seaborn** – Visualization
- **Pandas / NumPy** – Data manipulation

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook breast_cancer.ipynb
   ```

---

## ✨ Author

**Cristhian Zambrano**  
Master’s in Data Science | Data Analyst | Deep Learning Enthusiast  
[LinkedIn](https://www.linkedin.com/in/cristhianzambrano) | [GitHub](https://github.com/cristhian2900)