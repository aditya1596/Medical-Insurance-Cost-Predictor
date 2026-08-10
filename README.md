# Medical Insurance Cost Prediction

An end-to-end Machine Learning project that predicts individual medical insurance charges using demographic and lifestyle-related features. The project covers data analysis, preprocessing, multiple regression models, model comparison, feature importance, and deployment using Flask and Docker.

---

## Problem Statement

Medical insurance charges vary significantly based on factors such as age, BMI, smoking status, number of children, sex, and region.

The objective of this project is to build a regression model that can estimate an individual's medical insurance charges based on these demographic and lifestyle characteristics.

---

## Tech Stack

* **Python** – Data processing and Machine Learning
* **Pandas & NumPy** – Data manipulation and numerical computation
* **Matplotlib & Seaborn** – Exploratory Data Analysis and visualization
* **Scikit-learn** – Machine Learning models and preprocessing
* **XGBoost** – Gradient boosting model
* **Flask** – Model deployment and web application
* **Docker** – Containerization

---

## Dataset

**Source:** Kaggle – Medical Cost Personal Dataset

* **Records:** 1,338
* **Input Features:** 6
* **Target:** Medical insurance charges

### Features

| Feature  | Description                  | Type        |
| -------- | ---------------------------- | ----------- |
| age      | Age of the individual        | Numerical   |
| sex      | Gender                       | Categorical |
| bmi      | Body Mass Index              | Numerical   |
| children | Number of dependent children | Numerical   |
| smoker   | Smoking status               | Categorical |
| region   | Residential region           | Categorical |
| charges  | Medical insurance cost       | Target      |

---

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Exploratory Data Analysis
     ↓
Categorical Encoding
     ↓
Target Transformation
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Multiple Regression Models
     ↓
Model Evaluation
     ↓
Feature Importance
     ↓
Model Serialization
     ↓
Flask Deployment
     ↓
Docker Containerization
```

---

## Exploratory Data Analysis

The dataset was analyzed to understand:

* Age distribution
* Gender distribution
* BMI distribution
* Number of children
* Smoking status
* Regional distribution
* Insurance charge distribution

### Key observations

* The average age in the dataset is approximately 40 years.
* BMI has an average of approximately 30.66.
* There are 274 smokers out of 1,338 individuals.
* Insurance charges are strongly right-skewed.
* High insurance charges are particularly associated with smoking and higher age/BMI.

---

## Target Transformation

The original insurance charges showed strong right skewness with high-value outliers.

To reduce the effect of extreme values, the target variable was transformed using:

```python
df['log_charges'] = np.log1p(df['charges'])
```

The transformed target was then used for model training.

After prediction, the values were converted back to the original dollar scale using:

```python
np.expm1(predictions)
```

This allows the model to train on a more stable target distribution while reporting predictions in actual insurance-cost units.

---

## Data Preprocessing

### Categorical Encoding

The categorical variables were converted into numerical values:

```text
sex
male   → 0
female → 1

smoker
no  → 0
yes → 1

region
southeast → 0
southwest → 1
northeast → 2
northwest → 3
```

### Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

with:

```python
random_state = 42
```

### Feature Scaling

`StandardScaler` was used for models that are sensitive to feature scale:

* Linear Regression
* SVR
* KNN

Tree-based models were trained using the original numerical feature values.

---

## Models Compared

The following regression models were trained and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
6. Support Vector Regression
7. K-Nearest Neighbors Regressor

The models were evaluated using:

* **MAE**
* **MSE**
* **RMSE**
* **R² Score**

---

## Model Performance

| Model             |   MAE ($) |  RMSE ($) |        R² |
| ----------------- | --------: | --------: | --------: |
| **Random Forest** | **2,045** | **4,354** | **0.878** |
| Gradient Boosting |     2,002 |     4,410 |     0.875 |
| KNN               |     2,644 |     4,841 |     0.849 |
| SVR               |     2,210 |     4,928 |     0.844 |
| XGBoost           |     2,614 |     5,446 |     0.809 |
| Linear Regression |     3,925 |     7,863 |     0.602 |
| Decision Tree     |     2,630 |     5,947 |     0.772 |

### Final Model

**Random Forest Regressor** was selected as the final model because it achieved the highest R² score among the evaluated models.

Final configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=7,
    random_state=42
)
```

The final model achieved an **R² of approximately 0.878** on the test data.

This means the model explains approximately **87.8% of the variance in insurance charges** on the evaluated test set.

---

## Feature Importance

Feature importance from the final Random Forest model showed the following approximate ranking:

1. **Smoker**
2. **Age**
3. **BMI**
4. **Children**
5. **Region**
6. **Sex**

The results indicate that smoking status and age are the strongest predictive features in the model.

Feature importance should be interpreted as model-based predictive importance and should not be treated as proof of causality.

---

## Deployment

The trained Random Forest model was serialized using Python's `pickle` library:

```python
pickle.dump(random_forest_model, open('model.pkl', 'wb'))
```

The preprocessing scaler was also saved:

```python
pickle.dump(sr, open('scaler.pkl', 'wb'))
```

A Flask application was created to accept user inputs and generate insurance-cost predictions.

The application accepts:

* Age
* Sex
* BMI
* Number of children
* Smoking status
* Region

The prediction is then converted back from log scale using `np.expm1()` and displayed as the estimated insurance cost.

---

## Flask API

The application contains two main routes:

### Home Route

```text
/
```

Displays the input form.

### Prediction Route

```text
/predict
```

Accepts user inputs through a POST request and generates the predicted insurance charge.

---

## Dockerization

The application was containerized using Docker to make the deployment environment reproducible and portable.

### Project Structure

```text
medical-insurance-cost-prediction/
│
├── Medical_insurence_cost_prediction.ipynb
├── app.py
├── Dockerfile
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── insurance.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── ...
```

---

## Key Insights

* **Smoking status** is the strongest predictive feature in the final model.
* **Age** is another major contributor to predicted insurance charges.
* Higher BMI is associated with higher insurance costs.
* Insurance charges have a strongly right-skewed distribution.
* Tree-based ensemble models performed substantially better than the basic Linear Regression model on this dataset.
* Random Forest achieved the highest R² among the tested models.

---

## Conclusion

This project demonstrates a complete Machine Learning workflow for predicting medical insurance charges, from exploratory data analysis and preprocessing to model comparison and deployment.

Among the evaluated models, **Random Forest Regressor achieved the highest R² score of approximately 0.878**, making it the selected model for deployment.

