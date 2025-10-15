#  JSON-Driven AutoML Pipeline
## A Configurable Framework for Dynamic ML Model Training and Evaluation

This project implements a **fully automated, end-to-end machine learning pipeline** built from scratch using **Python** and **Scikit-learn**. The entire workflow—from data preprocessing to model tuning and evaluation—is controlled by a single JSON configuration file, requiring **zero code changes** to run new experiments.

Created By-
### **Pranit Gore**
### **Contact Me- ***pranitgore05@gmail.com***
---

## ✨ Core Features & Key Advantages

| Feature | Description |
| :--- | :--- |
| **JSON-Driven Workflow** | Change models, features, and parameters by editing `algoparams.json`. **No Python code modification is needed.** |
| **Dynamic Preprocessing** | Automatically handles **numerical** (imputation) and **categorical** (encoding) columns using `ColumnTransformer`. |
| **Automated Tuning** | Uses **GridSearchCV** for automated, efficient hyperparameter search. |
| **Model & Task Agnostic** | Works seamlessly for both **Regression** and **Classification** tasks. |
| **Feature Reduction** | Configurable support for **PCA** or **Tree-based** feature selection. |
| **Clear Evaluation** | Automatically selects and reports the correct metrics for the defined task. |

---

## 🛠️ Pipeline Architecture (How It Works)

The script executes a series of steps based entirely on the settings within the `algoparams.json` file.

### 1. 📥 Configuration & Data Loading

* Loads the dataset (e.g., `iris.csv`).
* The JSON file defines the **target column** and the **prediction type** (e.g., `"regression"`).
* Reads all design parameters, including feature lists, reduction method, and algorithm-specific hyperparameter ranges.

### 2. ⚙️ Automated Preprocessing

Features are handled dynamically based on their type:

* **Numerical:** Applies `SimpleImputer(strategy="mean")`.
* **Categorical:** Applies `OneHotEncoder(handle_unknown="ignore")`.

These steps are combined using a `ColumnTransformer` for simultaneous, robust processing.

### 3. 📉 Feature Reduction (Optional)

Configure dimensionality reduction in the JSON file using one of three options:

| Method | Description |
| :--- | :--- |
| **`PCA`** | Principal Component Analysis (Dimensionality Reduction). |
| **`Tree-based`** | Selects important features using an `ExtraTrees` model. |
| **`None`** | Keeps all original features (`passthrough`). |

### 4.  Dynamic Model Selection & Tuning

1.  The script reads all algorithms marked `"is_selected": true` from the JSON.
2.  A complete `Pipeline` is built for each model: `[Preprocess, Feature Reduction, Model]`.
3.  **Grid Search:** `GridSearchCV` is executed against the full pipeline using the hyperparameter ranges provided in the JSON.

> **Supported Models (Example):**
> * **Regression:** `RandomForestRegressor`, `Ridge`, `Lasso`, `SVR`.
> * **Classification:** `RandomForestClassifier`, `SVC`.

### 5. Metric Evaluation

The best-performing model is evaluated, and the results are printed:

* **Regression:** $R^2$, MAE, MSE
* **Classification:** Accuracy, F1 Score

---

##  Example JSON Configuration

This snippet sets up a **Regression** task using **PCA** and tunes a **RandomForestRegressor** with a defined range of trees and depth.

```json
{
  "design_state_data": {
    "target": {
      "target": "petal_width",
      "prediction_type": "regression"
    },
    "feature_reduction": {
      "feature_reduction_method": "PCA",
      "num_of_features_to_keep": 2
    },
    "algorithms": {
      "RandomForestRegressor": {
        "is_selected": true,
        "min_trees": 10,
        "max_trees": 20,
        "min_depth": 5,
        "max_depth": 25
      }
    }
  }
}



