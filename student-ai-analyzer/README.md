# Student AI Analyzer

## Student Performance & Career Prediction using Machine Learning

> **AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026**

---

## 📌 Project Objective

Build an end-to-end Machine Learning web application that demonstrates the core concepts covered in the AICTE | IBM SkillsBuild ML & Applied AI Internship — including classification, regression, clustering, deep learning fundamentals, and reinforcement learning.

---

## ❓ Problem Statement

Students often struggle to identify a suitable career path in the technology industry. This application uses a student's academic performance and skill profile to suggest a career category using Machine Learning models.

---

## ✨ Features

- **Career Prediction** – Decision Tree Classifier predicts a suitable tech career
- **Salary Prediction** – Linear Regression predicts expected salary from experience
- **Student Clustering** – K-Means groups students by performance level
- **Deep Learning Demo** – Small neural network using TensorFlow/Keras (optional)
- **Q-Learning Demo** – Simple reinforcement learning simulation
- **Interactive UI** – Built with Streamlit, no HTML/CSS knowledge needed

---

## 🛠️ Technologies Used

| Technology   | Purpose                              |
|--------------|--------------------------------------|
| Python 3.9+  | Core programming language            |
| Pandas       | Data manipulation and analysis       |
| NumPy        | Numerical computations               |
| Scikit-learn | ML algorithms                        |
| Plotly       | Interactive visualisations           |
| Matplotlib   | Static plots                         |
| Streamlit    | Web application framework            |
| TensorFlow   | Deep Learning demo *(optional)*      |

---

## 🤖 Machine Learning Concepts Demonstrated

1. Machine Learning fundamentals
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Supervised Learning
5. Regression (Linear Regression)
6. Classification (Decision Tree)
7. Decision Trees (Gini impurity, feature importance)
8. Unsupervised Learning
9. K-Means Clustering
10. Basic Deep Learning (Neural Network with Keras)
11. Reinforcement Learning / Q-Learning concepts
12. Model evaluation (accuracy, confusion matrix, MAE, RMSE, R²)
13. Interactive ML application (Streamlit)

---

## 🧠 Algorithms Used

| Algorithm              | Type            | Used In                  |
|------------------------|-----------------|--------------------------|
| Decision Tree          | Classification  | Career Prediction page   |
| Linear Regression      | Regression      | Salary Prediction page   |
| K-Means Clustering     | Unsupervised    | Student Clustering page  |
| Neural Network (Keras) | Deep Learning   | Deep Learning Basics page|
| Q-Learning             | Reinforcement   | Q-Learning Demo page     |

---

## 📂 Dataset Information

> ⚠️ **Important Notice:** Both datasets used in this project are **synthetically generated** for educational demonstration purposes. They do **NOT** contain data from real students. The patterns in the data are crafted to make the ML workflow clearly visible and understandable.

| File                   | Description                          | Rows |
|------------------------|--------------------------------------|------|
| `data/student_data.csv`| Student academic & skill features    | 200  |
| `data/salary_data.csv` | Years of experience and salary       | 60   |

### student_data.csv columns

| Column               | Description                          |
|----------------------|--------------------------------------|
| StudyHours           | Hours studied per day (0–12)         |
| Attendance           | Attendance percentage (0–100)        |
| PreviousMarks        | Previous exam score (0–100)          |
| PythonSkill          | Python proficiency (1–10)            |
| SQLSkill             | SQL proficiency (1–10)               |
| MLSkill              | Machine Learning skill (1–10)        |
| CommunicationSkill   | Communication ability (1–10)         |
| ProblemSolvingSkill  | Problem-solving ability (1–10)       |
| Career               | Target label (4 career categories)   |

**Career categories:** Data Analyst, Software Developer, ML Engineer, Web Developer

---

## 📁 Project Structure

```
student-ai-analyzer/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore
│
├── data/
│   ├── student_data.csv    ← Synthetic student dataset (200 rows)
│   └── salary_data.csv     ← Synthetic salary dataset (60 rows)
│
├── models/
│   ├── classifier.py       ← Decision Tree Classifier
│   ├── regression.py       ← Linear Regression
│   ├── clustering.py       ← K-Means Clustering
│   ├── deep_learning.py    ← Neural Network demo (TensorFlow)
│   └── q_learning.py       ← Q-Learning simulation
│
└── utils/
    └── preprocessing.py    ← Data loading & preprocessing helpers
```

---

## 🔄 Project Workflow

```
Dataset (CSV)
    ↓
Data Cleaning (drop nulls, type checks)
    ↓
Exploratory Data Analysis
    ↓
Feature Selection
    ↓
Train / Test Split (80/20, random_state=42)
    ↓
Model Training (Decision Tree, LinearRegression, KMeans, etc.)
    ↓
Model Evaluation (accuracy, confusion matrix, MAE, RMSE, R²)
    ↓
Prediction (user input via Streamlit sliders)
```

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/student-ai-analyzer.git
cd student-ai-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install TensorFlow for the Deep Learning page

```bash
pip install tensorflow
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 🖥️ Example Usage

1. Open the app in your browser.
2. Use the **sidebar** to navigate between pages.
3. On the **Career Prediction** page, adjust the sliders and click **Predict Career**.
4. On the **Salary Prediction** page, enter years of experience and click **Predict Salary**.
5. On the **K-Means Clustering** page, explore how students are grouped.
6. On the **Deep Learning Basics** page, click the button to train a small neural network.
7. On the **Q-Learning Demo** page, run the simulation to see the agent learn.

---

## ⚠️ Limitations

- The dataset is entirely synthetic, so predictions are for demonstration only.
- The neural network module requires TensorFlow; the rest of the app works without it.
- Only 4 career categories are included (simplified for educational clarity).
- No real student data was collected or validated.

---

## 🚀 Future Improvements

- Collect real (anonymised) student data for genuine generalisation.
- Add more career categories and richer features.
- Try Random Forest or XGBoost for improved accuracy.
- Add a data upload feature so users can train on their own CSV.
- Deploy to Streamlit Cloud or Hugging Face Spaces.
- Add cross-validation for more robust evaluation.

---

## 📄 License

This project is for educational purposes. Feel free to fork, modify, and use it for learning.

---

*Built as part of the AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026.*
