"""MLflow Project entry point (Kriteria 3): re-train model saat CI dipantik.

Logging ke MLflow file store lokal (./mlruns) sehingga bisa jalan di CI tanpa
server. Menyimpan model + metrik, lalu mencetak run_id agar step berikutnya
(build-docker / upload artefak) bisa menemukannya.
"""
import argparse
import json

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "Churn"


def main(data_dir, n_estimators, max_depth):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    mlflow.set_experiment("telco_churn_ci")
    with mlflow.start_run(run_name="ci_retrain") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=(None if max_depth < 0 else max_depth),
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        mlflow.sklearn.log_model(model, artifact_path="model")

        info = {"run_id": run.info.run_id, "model_uri": f"runs:/{run.info.run_id}/model"}
        with open("run_info.json", "w") as f:
            json.dump(info, f)
        print("RUN_INFO", json.dumps(info))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="telco_churn_preprocessing")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=-1)  # -1 => None
    a = ap.parse_args()
    main(a.data_dir, a.n_estimators, a.max_depth)
