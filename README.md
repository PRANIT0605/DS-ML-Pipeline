# AutoML JSON-Driven Model Builder
## A generic, configurable pipeline for dynamic ML model training and evaluation
### **Overview**

This project implements a fully automated, JSON-driven machine learning pipeline built entirely from scratch using Python and Scikit-learn.

It allows users to:
- Load any dataset (e.g., iris.csv)
- Define all pipeline settings in a single JSON configuration file (algoparams.json)

Automatically:

- Handle features (categorical & numerical)
- Apply imputations and encodings
- Perform feature reduction (PCA / Tree-based)
- Select and tune ML models dynamically
- Evaluate metrics for regression or classification tasks in one go

In short, you only modify the JSON file, and this script does everything else automatically — preprocessing, model tuning, and metric computation.

## **Core Features**

- JSON-Driven Workflow – No code changes needed. Modify the JSON file to change models, features, or parameters.
- Dynamic Preprocessing – Automatically detects numerical and categorical columns and applies imputers and encoders.
- Feature Reduction – Supports PCA, Tree-based feature selection, or None.
- Model Selection – Chooses models (RandomForest, Ridge, Lasso, SVM, etc.) based on JSON configuration.
- Hyperparameter Tuning – Uses GridSearchCV for automated best-parameter selection.
- Auto Metric Evaluation – Chooses the right metrics:

Regression → R², MAE, MSE

Classification → Accuracy, F1 Score

## Pipeline Architecture
Step 1: Dataset Loading
data = pd.read_csv("iris.csv")

- Loads the dataset dynamically (you can replace with any dataset).
- JSON defines which column is the target variable.

Step 2: Configuration Loading
with open("algoparams.json") as f:
    config = json.load(f)["design_state_data"]


Reads pipeline design parameters from the JSON file.

Includes information such as:
- Target column & prediction type
- Feature handling (categorical/numerical)
- Feature reduction technique
- Algorithms & their hyperparameter ranges

Step 3: Feature Handling

Automatically separates columns:

- Numerical → Uses SimpleImputer(strategy="mean")
- Categorical → Uses OneHotEncoder(handle_unknown="ignore")
These are combined using a ColumnTransformer, ensuring both types are processed in parallel.

Step 4: Feature Reduction

Supports three configurable reduction methods:
- PCA → Dimensionality reduction via principal components
- Tree-based → Selects important features using ExtraTrees models
- None / passthrough → Keeps all features

Defined dynamically by:

feature_reduction_method: "PCA" or "Tree-based" or "None"
in the JSON file.

Step 5: Model Selection

- The function get_models() dynamically reads selected algorithms from the JSON file:
- Each algorithm includes its tuning parameters and flags like "is_selected": true
- Builds a tuple list: (model_name, model_object, param_grid)

Supported models:
- Regression: RandomForestRegressor, LinearRegression, Ridge, Lasso, DecisionTree, SVR
- Classification: RandomForestClassifier, SVC
- Each model’s hyperparameters are directly extracted from the JSON (e.g., number of trees, max depth, regularization strength, etc.).

Step 6: Grid Search and Evaluation

For each model:
Builds a complete pipeline:

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("reduce", feature_reduction_step),
    ("model", model)
])


### Runs a grid search:
grid = GridSearchCV(pipe, param_grid, cv=3, scoring=scoring_metric)


### Evaluates performance:
- Regression: R², MAE, MSE
- Classification: Accuracy, F1-score
- Prints best parameters and metrics:

Running model: RandomForestRegressor
Best Params: {'model__max_depth': 20, 'model__n_estimators': 50}
R2: 0.89
MAE: 0.16
MSE: 0.05
------------------------------------------------------------

##  Example JSON Configuration
{
  "design_state_data": {
    "target": {
      "target": "petal_width",
      "prediction_type": "regression"
    },
    "feature_handling": {
      "sepal_length": {"is_selected": true, "feature_variable_type": "numerical"},
      "sepal_width": {"is_selected": true, "feature_variable_type": "numerical"},
      "petal_length": {"is_selected": true, "feature_variable_type": "numerical"}
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
        "max_depth": 25,
        "min_samples_per_leaf_min_value": 1,
        "min_samples_per_leaf_max_value": 5
      }
    }
  }
}

## Output Example
Target Column: petal_width | Prediction Type: regression
Dataset Shape: (150, 5)

## Running model: RandomForestRegressor
Best Params: {'model__max_depth': 20, 'model__min_samples_leaf': 5, 'model__n_estimators': 20}
R2: 0.793
MAE: 0.293
MSE: 0.132
------------------------------------------------------------

## Key Advantages

- Generic Framework: Works for both regression and classification tasks.
- Modular Design: Every stage (preprocessing, feature selection, modeling) can be swapped independently.
- Human-Readable Config: Non-developers can control experiments via JSON.
- Fast Experimentation: Change models and rerun instantly — no code changes required.
- Scalable: Easily extendable to include new algorithms or feature engineering blocks.

## How to Run

- Place your dataset (e.g., iris.csv) in the same directory.
- Configure your parameters in algoparams.json.

Run:
## **python main.py**

Observe metrics for all selected models printed to console.

## Future Improvements

- Add Auto-save results to CSV/Excel.
- Integrate visualization dashboards (e.g., Power BI or Streamlit).
- Extend to neural networks or XGBoost/LightGBM.
- Include automatic feature scaling options per JSON.

## Conclusion

This project demonstrates how to create a generic AutoML system without relying on third-party frameworks.
It is ideal for:

- Experimentation pipelines
- Automated ML assignment evaluation

- Research & benchmarking setups
