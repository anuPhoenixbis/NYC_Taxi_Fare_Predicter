# NYC Taxi Fare Predictor

## Project Overview

This machine learning project predicts New York City taxi fares using location coordinates, passenger count, and datetime information. The model was developed as part of the [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) Kaggle competition.

## Key Features

- **Data Cleaning & Preprocessing**: Handling outliers, invalid coordinates, and ensuring data quality
- **Feature Engineering**: 
  - Created distance features using the Haversine formula
  - Added landmark distance features (e.g., airports, tourist attractions)
  - Extracted date/time components for temporal analysis
- **Model Development**: 
  - Built and compared multiple regression models
  - Implemented hyperparameter tuning for XGBoost
  - Achieved RMSE of ~3.9 on validation data

## Technical Approach

### Data Preprocessing
- Sampled 1% of the massive dataset for exploration and model development
- Removed invalid coordinates, passenger counts, and fare amounts
- Limited coordinates to the NYC area
- Extracted temporal features from pickup datetime

### Feature Engineering
- Calculated trip distance using the Haversine formula
- Created features measuring distance to key NYC landmarks:
  - JFK, LaGuardia, and Newark airports
  - Times Square, Empire State Building, Central Park
  - World Trade Center, Statue of Liberty, Brooklyn Bridge
  - Metropolitan Museum of Art

### Model Development & Evaluation
- **Baseline Models**:
  - Mean Regressor (RMSE: ~9.0)
  - Linear Regression (RMSE: ~9.0)
- **Advanced Models**:
  - Ridge Regression (RMSE: ~4.9)
  - Random Forest Regressor
  - XGBoost Regressor (RMSE: ~3.9)
  - Lasso Regression

### Hyperparameter Tuning
- Tuned XGBoost parameters:
  - n_estimators: 500
  - max_depth: 5
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8

## Technologies Used

- **Python**: Primary programming language
- **Libraries**:
  - pandas & numpy: Data manipulation
  - scikit-learn: Machine learning models
  - XGBoost: Gradient boosting implementation
  - matplotlib & seaborn: Data visualization
  - plotly: Interactive visualizations
- **Environment**: Google Colab

## Setup & Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nyc-taxi-fare-predictor.git
   ```

2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost
   ```

3. Download the [competition dataset](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)

4. Run the notebook or Python script:
   ```
   jupyter notebook NYC_Taxi_Fare_Predicter.ipynb
   ```

## Results

- Successfully reduced RMSE from 9.0 (baseline) to 3.9 (tuned XGBoost)
- Identified key features impacting taxi fare prediction:
  - Trip distance (most important)
  - Proximity to major landmarks
  - Time-based features (hour of day, day of week)

## Future Improvements

- Incorporate weather data as additional features
- Add traffic congestion metrics
- Implement neural network models for comparison
- Optimize for both prediction accuracy and inference speed
