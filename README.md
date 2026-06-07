## Medical Insurance Cost Prediction

End-to-end ML pipeline to predict individual insurance charges based on demographic and health risk factors — with Flask API deployment and Docker containerization.

---
## Problem Statement
Insurance companies need to estimate medical charges accurately to price premiums fairly. Overestimation leads to customer loss; underestimation leads to financial risk. This project builds a predictive model using patient demographic and lifestyle data to estimate annual insurance charges


---

##  Technologies Used

* **Python** – Data processing and model building
* **Scikit-learn** – Regression models
* **Pandas, NumPy** – Data handling
* **Matplotlib, Seaborn** – Visualization
* **Flask** – Web application deployment
* **Docker** – Containerization

---
## Dataset

 Source: Kaggle — Medical Cost Personal Dataset
 Size: 1,338 records, 6 features (age, sex, BMI, children, smoker, region)
 Target: Annual insurance charges (USD)


##  ML Workflow

* Data cleaning and preprocessing
* Feature encoding (categorical variables)
* Model training using:

  * Linear Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting 
  * KNN
  * XGBoost
  * SVR
* Model evaluation using MAE, RMSE, and R²

---
## Models Compared
| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| **Gradient Boosting** ✅ | 2,403 | 4,319 | **0.8798** |
| Random Forest | 2,495 | 4,462 | 0.8717 |
| XGBoost | 2,719 | 4,912 | 0.8446 |
| KNN | 3,079 | 5,093 | 0.8329 |
| Linear Regression | 4,190 | 5,803 | 0.7830 |
| Decision Tree | 3,350 | 7,104 | 0.6748 |
| SVR | 8,600 | 12,881 | -0.069 |

Gradient Boosting performed best — outperformed Linear Regression by **12.3% in R²**.

## Results

- Best model: Gradient Boosting Regressor
- R² Score: 0.8798 (model explains ~88% variance in insurance charges)
- GB outperformed Linear Regression by 12.3% in R² (0.8798 vs 0.7830)
- Deployed as real-time REST API via Flask, containerized with Docker

---

##  Deployment

* Built a Flask-based web app (`app.py`) for user interaction
* Users can input details and get predicted insurance cost
* Dockerized application for scalable deployment

---

##  Project Structure

```insurance-risk-ml/
│
├── Medical_insurence_cost_prediction.ipynb  # Full ML pipeline
├── app.py                                   # Flask API
├── Dockerfile                               # Docker config
├── requirements.txt                         # Dependencies
├── model.pkl                                # Trained GB model
├── scaler.pkl                               # StandardScaler
├── insurance.csv                            # Dataset
├── templates/                               # HTML templates
└── static/                                  # CSS/JS assets/
```

---

 ## Key Insights

- Smokers incur 3-4x higher charges regardless of age — strongest single risk factor
- BMI > 30 combined with smoking creates a compounding risk multiplier
- Age has a near-linear relationship with charges for non-smokers
- Region has minimal predictive power — insurers can deprioritize geographic segmentation

---

##  Contact

**Aditya Singh Shekhawat | IIT Indore**

 * GitHub: https://github.com/aditya1596
* LinkedIn: https://www.linkedin.com/in/aditya-singh-shekhawat-934930288/

