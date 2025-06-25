#  Fertilizer Prediction - Kaggle Playground Series S5E6

Welcome to my solution for the **[Kaggle Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6/overview)**.  
In this competition, the goal was to predict the correct **fertilizer name** given a set of soil and crop characteristics.

I participated in this challenge to enhance my skills in feature engineering, model evaluation, and hyperparameter optimization.  
My final submission achieved a **MAP@3 score of `0.XXXX`** on the private leaderboard.

## 🔍 Problem Overview

The dataset contains various soil properties and environmental conditions.  
The task is a **multi-class classification** problem, where the model must predict the top 3 most likely fertilizer types in ranked order.

### 🧮 Evaluation Metric: MAP@3

The evaluation metric for this competition is **Mean Average Precision at k = 3 (MAP@3)**.

This metric evaluates how well the model ranks the true class within its top 3 predictions.  
The earlier the correct class appears in the prediction list, the higher the score.  
For example, predicting the correct fertilizer at rank 1 gives full credit, while at rank 2 or 3 gives partial credit.  
If the correct class is not in the top 3 predictions, the model gets zero credit for that instance.

It is especially useful in multi-class problems where **ranking predictions** is more important than selecting just one label.

Dataset
---
The dataset consists of various numerical and categorical features related to soil properties and agricultural context. Below is a brief description of each feature and what it represents:

* **`Temperature`**: A numerical variable representing the ambient temperature in degrees Celsius. 

* **`Humidity`**: The relative humidity percentage in the environment.

* **`Moisture`**: Displays the soil moisture content as a numerical percentage. Helps assess how wet or dry the soil is.

* **`Soil Type`**: A categorical attribute indicating the general type of soil (e.g., sandy, loamy, clayey). 

* **`Crop Type`**: A categorical variable that defines the type of crop grown in the field. 

* **`Nitrogen`**: A numerical value indicating the nitrogen content of the soil. 

* **`Phosphorus`**: The phosphorus content of the soil, measured as a numerical value. 

* **`Potassium`**: Indicates the amount of potassium present in the soil.

* **`Fertiliser Name`**: This is the target variable. It specifies the recommended fertiliser based on the given conditions. It is a categorical label used during training.


---

## ⛓️ Project Structure

The project primarily consists of two Python files:
- `main.py`: Contains data loading, exploratory data analysis (EDA), feature engineering, modeling, hyperparameter optimization, and prediction generation steps.
- `preprocessing.py`: A separate module called within `main.py`, containing data preprocessing steps (feature engineering, encoding, scaling). This structure ensures that consistent transformations are applied to both the training and test datasets.


## 🛠️ Methods & Workflow

Throughout this notebook, I followed a structured machine learning pipeline:

### 1. Data Preprocessing
- Handling missing values and type conversions
- Label Encoding for categorical variables
- Feature scaling (if required)

### 2. Feature Engineering
- Created additional domain-specific features (e.g. `NEW_...`)
- Removed less informative features based on feature importance

### 3. Model Training & Evaluation
I trained and compared several ensemble-based models:
- **XGBoost**
- **CatBoost**
- **LightGBM**

The best performing model (XGBoost) was selected based on **MAP@3** score on the validation set.

### 4. Hyperparameter Optimization
- Used **Optuna** to tune hyperparameters efficiently
- Applied **early stopping** to reduce overfitting
- Focused on maximizing **MAP@3** rather than traditional accuracy


### 5. Submission Strategy
- Extracted top 3 predictions using `predict_proba()` + `argsort()`
- Inverse transformed label encodings to match submission format
- Saved the final model using `joblib` for reuse

---

## Setup and Running

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/BahriDogru/Predicting_Optimal_Fertilizers.git
    ```
2.  **After cloning, go to the project directory**:
    ```bash
    cd Predicting_Optimal_Fertilizers
    ```
3. **Create and activate the Conda environment**:
    All libraries required for the project are listed in the `environment.yaml` file.
    You can use this file to automatically create and activate the Conda environment.
    ```bash
    conda env create -f environment.yaml
    conda activate predicting_fertilizers_env
    ```
4. **Download the Dataset**:
    Download the `train.csv` and `test.csv` files from the Kaggle competition page ([https://www.kaggle.com/competitions/playground-series-s5e5](https://www.kaggle.com/competitions/playground-series-s5e5)). Place these files inside a folder named `dataset/` in your project's root directory.
    ```
    .
    ├── main.py
    ├── preprocessing.py
    ├── dataset/
    │   ├── train.csv
    │   └── test.csv
    └── README.md
    ```
5. **Run the Code**:
    ```bash
    python main.py
    ```
    This command will train the model, make predictions, and generate the `submission.csv` file.


#### You can access the Kaggle Notebook I prepared for this competition via this [link](https://www.kaggle.com/code/bahridgr/fertilizer-prediction-kaggle-competition)