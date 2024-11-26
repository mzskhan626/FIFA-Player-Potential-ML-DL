# FIFA-Player-Potential-ML-DL


## **FIFA Player Potential Prediction**

### **Overview**
This project utilizes machine learning and deep learning techniques to predict FIFA player potential. By analyzing key attributes like physical characteristics, skill metrics, and performance ratings, the project provides insights into player potential, enabling better decision-making for football clubs.

---

## **Project Objectives**
- Predict FIFA player potential using:
  1. **Random Forest Regressor** (Machine Learning).
  2. **Deep Neural Network** (Deep Learning).
- Provide insights into key features that influence player potential.
- Streamline the scouting process for football clubs by offering a data-driven approach.

---

## **Dataset**
- **Source**: FIFA dataset containing 17,954 rows and 51 columns.
- **Key Features**:
  - Physical attributes: `age`, `height_cm`, `weight_kgs`.
  - Skill metrics: `dribbling`, `ball_control`, `crossing`.
  - Performance metrics: `overall_rating`, `potential`.
- **Preprocessing**:
  - Missing values handled.
  - Categorical variables encoded.
  - Numerical features scaled.

---

## **Software and Tools Used**

### **1. Python Libraries**
1. **pandas**: For data manipulation, preprocessing, and analysis.  
2. **numpy**: For numerical computations and array handling.  
3. **matplotlib**: For creating static, animated, and interactive visualizations.  
4. **seaborn**: For statistical data visualization, making it easy to generate attractive and informative plots.  
5. **scikit-learn**: For implementing machine learning models, feature scaling, and model evaluation.  
6. **tensorflow** and **keras**: For building, training, and fine-tuning deep learning models.  
7. **joblib**: For saving and loading machine learning models efficiently.  
8. **statsmodels**: For statistical computations and advanced data exploration.  
9. **xgboost**: For implementing boosted decision tree models to enhance comparisons.  
10. **plotly**: For creating interactive and dynamic visualizations for exploratory data analysis.  

### **2. Development Tools**
1. **Jupyter Notebook**: For developing, documenting, and running the project interactively.  
2. **VS Code**: For additional code editing and debugging.  

### **3. Deep Learning Tools**
1. **TensorBoard**: For visualizing and monitoring the deep learning model training process.  
2. **Optuna**: For advanced hyperparameter tuning to optimize model performance.  

### **4. Environment**
1. **Python Version**: 3.8+  
2. **IDE**: Jupyter Notebook and VS Code  
3. **Operating System**: Windows/Linux/MacOS (cross-platform compatibility)  

---

## **Models Used**

### **1. Random Forest Regressor**
- Justification: Robust to overfitting and effective for non-linear regression.
- Metrics: Mean Squared Error (MSE), \( R^2 \) Score.
- Results: Achieved an \( R^2 \) score of ~0.94.

### **2. Deep Neural Network**
- Architecture:
  - Input Layer: All player features.
  - Hidden Layers: Fully connected layers with ReLU activation and Dropout regularization.
  - Output Layer: Linear activation for regression.
- Metrics: Mean Absolute Error (MAE), Loss.
- Results: Achieved a low MAE of ~2.5.

---

## **Project Structure**
/fifa-player-potential
├── fifa_players.csv         # Dataset used for the project
├── final_project.ipynb      # Jupyter Notebook containing all code and outputs
├── logs/
│   └── fit/                     # TensorBoard logs for deep learning
├── README.md                    # Project description and details


## **Results**
- **EDA Findings**:
  - Strong correlations between player potential and attributes like `value_euro`, `ball_control`, and `dribbling`.
- **Machine Learning**:
  - Random Forest achieved a high \( R^2 \) score of ~0.94.
- **Deep Learning**:
  - Neural Network performed well with a Mean Absolute Error (MAE) of ~2.5.

---
