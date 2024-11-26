# FIFA-Player-Potential-ML-DL
##FIFA Player Potential Prediction
###Overview
This project utilizes machine learning and deep learning techniques to predict FIFA player potential. By analyzing key attributes like physical characteristics, skill metrics, and performance ratings, the project provides insights into player potential, enabling better decision-making for football clubs.

###Project Objectives
Predict FIFA player potential using:
Random Forest Regressor (Machine Learning).
Deep Neural Network (Deep Learning).
Provide insights into key features that influence player potential.
Streamline the scouting process for football clubs by offering a data-driven approach.
Dataset
Source: FIFA dataset containing 17,954 rows and 51 columns.
Key Features:
Physical attributes: age, height_cm, weight_kgs.
Skill metrics: dribbling, ball_control, crossing.
Performance metrics: overall_rating, potential.
Preprocessing:
Missing values handled.
Categorical variables encoded.
Numerical features scaled.
Software and Tools Used
Python Libraries:
pandas and numpy: For data manipulation and analysis.
matplotlib and seaborn: For data visualization.
scikit-learn: For implementing machine learning models.
tensorflow and keras: For building and training deep learning models.
Jupyter Notebook: For developing and running the project interactively.
TensorBoard: For visualizing the deep learning model's training progress.
Git: For version control and collaboration.
Environment:
Python version: 3.8+
Development Environment: Jupyter Notebook and VS Code.
Models Used
1. Random Forest Regressor
Justification: Robust to overfitting and effective for non-linear regression.
Metrics: Mean Squared Error (MSE), 
𝑅
2
R 
2
  Score.
Results: Achieved an 
𝑅
2
R 
2
  score of ~0.94.
2. Deep Neural Network
Architecture:
Input Layer: All player features.
Hidden Layers: Fully connected layers with ReLU activation and Dropout regularization.
Output Layer: Linear activation for regression.
Metrics: Mean Absolute Error (MAE), Loss.
Results: Achieved a low MAE of ~2.5.
Project Structure
bash
Copy code
/fifa-player-potential
├── data/
│   └── fifa_players.csv         # Dataset used for the project
├── notebooks/
│   └── final_project.ipynb      # Jupyter Notebook containing all code and outputs
├── logs/
│   └── fit/                     # TensorBoard logs for deep learning
├── results/
│   ├── visualizations.png       # Key plots and visualizations
│   └── feature_importance.png   # Feature importance plot
├── README.md                    # Project description and details
└── requirements.txt             # List of required Python packages
Results
EDA Findings:
Strong correlations between player potential and attributes like value_euro, ball_control, and dribbling.
Machine Learning:
Random Forest achieved a high 
𝑅
2
R 
2
  score of ~0.94.
Deep Learning:
Neural Network performed well with a Mean Absolute Error (MAE) of ~2.5.
