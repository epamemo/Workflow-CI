# Workflow-CI — Epafraditus Memoriano (Kriteria 3)

Re-training model otomatis via **MLflow Project** + **GitHub Actions**.

## Struktur
| File | Keterangan |
|---|---|
| `MLProject/MLProject` | Definisi MLflow Project (entry point `main`) |
| `MLProject/conda.yaml` | Environment |
| `MLProject/modelling.py` | Training + logging model ke MLflow |
| `MLProject/telco_churn_preprocessing/` | Dataset hasil preprocessing |
| `.github/workflows/ci.yml` | CI: retrain → upload artefak → build & push Docker |

## Secrets yang harus diisi di GitHub (Settings → Secrets → Actions)
- `DOCKERHUB_USERNAME` — username Docker Hub
- `DOCKERHUB_TOKEN` — access token Docker Hub

## Menjalankan lokal
```bash
pip install mlflow==2.19.0 pandas scikit-learn numpy
mlflow run MLProject --env-manager=local -P n_estimators=300 -P max_depth=16
```

## Level: Advance
- [x] Basic — MLProject + CI membuat model saat trigger dipantik
- [x] Skilled — artefak model di-upload ke GitHub (`actions/upload-artifact`)
- [x] Advance — Docker image di-build (`mlflow models build-docker`) & push ke Docker Hub
