import pandas as pd
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score


data = pd.read_csv("iris.csv")
with open("algoparams.json") as f:
    config = json.load(f)["design_state_data"]

target_column = config["target"]["target"]
pred_type = config["target"]["prediction_type"].lower()
print(f"\nTarget Column: {target_column} | Prediction Type: {pred_type}")
print("Dataset Shape:", data.shape)

# Feature handling 
cat_col, num_col = [], []
for fname, fdetails in config["feature_handling"].items():
    if not fdetails["is_selected"]:
        continue
    if fname == target_column:  
        continue
    ftype = fdetails["feature_variable_type"]
    if ftype == "numerical":
        num_col.append(fname)
    elif ftype == "text":
        cat_col.append(fname)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_col),
    ("cat", categorical_transformer, cat_col)
])

#  Feature reduction 
def get_feature_reduction(config, pred_type):
    fr_config = config["feature_reduction"]
    method = fr_config["feature_reduction_method"]

    if method == "PCA":
        return PCA(n_components=int(fr_config["num_of_features_to_keep"]))
    elif method == "Tree-based":
        if pred_type == "regression":
            model = ExtraTreesRegressor(
                n_estimators=int(fr_config["num_of_trees"]),
                max_depth=int(fr_config["depth_of_trees"])
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=int(fr_config["num_of_trees"]),
                max_depth=int(fr_config["depth_of_trees"])
            )
        return SelectFromModel(model)
    else:
        return "passthrough"

feature_reduction_step = get_feature_reduction(config, pred_type)

# Model selection 
def get_models(config, pred_type):
    algos = config["algorithms"]
    selected_models = []

    for name, details in algos.items():
        if not details["is_selected"]:
            continue

        model, param_grid = None, {}

        if pred_type == "regression":
            if name == "RandomForestRegressor":
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    "model__n_estimators": [details["min_trees"], details["max_trees"]],
                    "model__max_depth": [details["min_depth"], details["max_depth"]],
                    "model__min_samples_leaf": [
                        details["min_samples_per_leaf_min_value"],
                        details["min_samples_per_leaf_max_value"]
                    ]
                }

            elif name == "LinearRegression":
                model = LinearRegression()
                param_grid = {}

            elif name == "RidgeRegression":
                model = Ridge()
                param_grid = {"model__alpha": [details["min_regparam"], details["max_regparam"]]}

            elif name == "LassoRegression":
                model = Lasso()
                param_grid = {"model__alpha": [details["min_regparam"], details["max_regparam"]]}

            elif name == "DecisionTreeRegressor":
                model = DecisionTreeRegressor()
                param_grid = {
                    "model__max_depth": [details["min_depth"], details["max_depth"]],
                    "model__min_samples_leaf": details.get("min_samples_per_leaf", [1])
                }

            elif name == "SVM":
                model = SVR()
                param_grid = {"model__C": details.get("c_value", [1])}

        elif pred_type == "classification":
            if name == "RandomForestClassifier":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    "model__n_estimators": [details["min_trees"], details["max_trees"]],
                    "model__max_depth": [details["min_depth"], details["max_depth"]]
                }
            elif name == "SVM":
                model = SVC()
                param_grid = {"model__C": details.get("c_value", [1])}

        if model is not None:
            selected_models.append((name, model, param_grid))

    return selected_models

selected_models = get_models(config, pred_type)
if not selected_models:
    print("\nNo models selected in JSON. Exiting.")
    exit()

# Train/Test Split
X = data.drop(columns=[target_column])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Training each model 
for model_name, model, param_grid in selected_models:
    print(f"\n Running model: {model_name}")
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("reduce", feature_reduction_step),
        ("model", model)
    ])

    scoring_metric = "r2" if pred_type == "regression" else "accuracy"
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring=scoring_metric)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    print("Best Params:", grid.best_params_)
    if pred_type == "regression":
        print("R2:", round(r2_score(y_test, y_pred), 3))
        print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
        print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
    else:
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
        print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted'), 3))
    print("-" * 60)

