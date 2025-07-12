# Advanced-Machine-Learning-for-House-Price-Prediction
This project used machine learning to improve house price prediction, a key concern for buyers, sellers, and investors. 

Using the Kaggle dataset "House Prices: Advanced Regression Techniques," which includes 79 variables in homes in Ames, Iowa, we applied data cleaning, normalization, and feature engineering to prepare the data. 
The work evaluated a wide range of property features, including physical attributes, location specifics, and neighborhood characteristics, to increase the accuracy of house price predictions. 
Regression models including Linear Regression, Random Forest, and Gradient Boosting were evaluated, along with hyperparameter tuning and cross-validation to optimize performance. 

# House Price Prediction API (FastAPI + XGBoost)

This project provides a live API for predicting house sale prices using a trained XGBoost regression model and engineered features, based on the Kaggle Ames housing dataset.

## Features
- Robust data cleaning and feature engineering pipeline
- Top feature selection (Pearson correlation)
- XGBoost model tuning (RandomizedSearchCV)
- API built with FastAPI for real-time prediction
- Example JSON input and output
- Input features:
  
| Feature          | Description                                   |
|------------------|-----------------------------------------------|
| OverallQual      | Overall material and finish quality (1–10)    |
| GrLivArea        | Above ground living area (square feet)        |
| GarageCars       | Number of cars the garage can hold            |
| TotalBsmtSF      | Total basement area (square feet)             |
| FullBath         | Number of full bathrooms                      |
| YearBuilt        | Year the house was built                      |
| YearRemodAdd     | Year of last remodel                          |
| MasVnrArea       | Masonry veneer area (square feet)             |
| Fireplaces       | Number of fireplaces                          |
| OpenPorchSF_log  | Log-transformed open porch area (sq ft)       |


## Sample API Input

```json
{
  "OverallQual": 7.0,
  "GrLivArea": 1710.0,
  "GarageCars": 2.0,
  "TotalBsmtSF": 856.0,
  "FullBath": 2.0,
  "YearBuilt": 2003.0,
  "YearRemodAdd": 2003.0,
  "MasVnrArea": 196.0,
  "Fireplaces": 0.0,
  "OpenPorchSF_log": 4.127134385045092
}
```

## Sample Output


``` json
{
  "predicted_sale_price": 199650.453125
}
```

## Usage

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Start the API server:
    ```
    uvicorn main:app --reload
    ```

3. Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger UI.


## Model Details


Model file: Models/tuned_xgb_model.pkl

API: FastAPI (Python 3.13+)

Credits
Developed by Stella Pang, 2025.


