import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("test-connection")

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("test", "success")
    print("✅ MLflow connection successful!")