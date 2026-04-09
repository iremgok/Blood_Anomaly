from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models(scale_pos_weight: float):

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=8,
            min_samples_split=16,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1),

        "SVM": SVC(
            kernel="rbf",
            C=0.5,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42) }
    return models
