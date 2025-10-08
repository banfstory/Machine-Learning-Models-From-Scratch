# Machine-Learning-Models-From-Scartch
### Description:
This project implements several machine learning models from scratch in Python, without using external ML libraries. Models included:
* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* K-Means Clustering
* Neural Network
* Gaussian Naive Bayes
* Multinomial Naive Bayes

### Tech Stack:
* Python – Core language for all model implementations
* NumPy / Pandas – Data handling and manipulation
* scikit-learn – Used only for splitting datasets into training and testing sets
* Matplotlib – Visualization of data or model results

## Datasets Used
1. Wisconsin Breast Cancer Database
* Used in: `gaussian_naive_bayes`, `k_nearest_neighbors`, `neural_network`
* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
* Citation:
   * O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.
   * William H. Wolberg and O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology", Proceedings of the National Academy of Sciences, U.S.A., Volume 87, December 1990, pp 9193-9196.
   * O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis", in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.
   * K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets", Optimization Methods and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).

2. Customer Purchasing Behaviors
* Used in: `linear_regression`
* Source: [Kaggle Dataset](https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors)
* Owner/Author: [Han Aksoy](https://www.kaggle.com/hanaksoy)
* License: CC0: Public Domain

## Getting Started / How to Run
***1. Create and activate a virtual environment***

***Windows PowerShell:***
```
python -m venv env
.\env\Scripts\Activate.ps1
```
***Windows Command Prompt (cmd.exe):***
```
python -m venv env
env\Scripts\activate.bat
```
***Mac/Linux:***
```
python3 -m venv env
source env/bin/activate
```
***2. Install required packages***
```
pip install -r requirements.txt
```
***3. Run a model script***
```
cd <model name>
python app.py
```

