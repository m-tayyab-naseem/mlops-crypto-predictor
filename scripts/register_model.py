"""
MLflow Model Registry Management
Register and manage model versions
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class ModelRegistry:
    """Manage models in MLflow Model Registry"""
    
    def __init__(self, tracking_uri=None):
        """Initialize MLflow client"""
        if tracking_uri is None:
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        print(f"✅ Connected to MLflow: {tracking_uri}")
    
    def list_experiments(self):
        """List all experiments"""
        experiments = self.client.search_experiments()
        
        print("\n📂 Available Experiments:")
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
        
        return experiments
    
    def get_best_run(self, experiment_name, metric='val_rmse', ascending=True):
        """Get the best run from an experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Search runs
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        best_run = runs[0]
        
        print(f"\n🏆 Best Run:")
        print(f"   Run ID: {best_run.info.run_id}")
        print(f"   Model: {best_run.data.params.get('model_type', 'Unknown')}")
        print(f"   {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
        print(f"   Run Name: {best_run.info.run_name}")
        
        return best_run
    
    def register_model(self, run_id, model_name, model_path="model"):
        """Register a model to Model Registry"""
        print(f"\n📝 Registering model...")
        
        # Get model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, model_name)
            
            print(f"Successfully registered model '{model_name}'.")
            print(f"   ✓ Version: {model_version.version}")
            print(f"   ✓ Run ID: {run_id}")
            
            return model_version
        
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a DagHub unsupported endpoint error
            if "unsupported endpoint" in error_str.lower() or "internal_error" in error_str.lower():
                print(f"   ⚠️  DagHub Model Registry limitation detected: {error_str}")
                print(f"   ℹ️  Attempting alternative registration method...")
                
                # Try alternative: use MlflowClient directly
                try:
                    # Check if model already exists
                    try:
                        existing_model = self.client.get_registered_model(model_name)
                        print(f"   ℹ️  Model '{model_name}' already exists in registry")
                    except:
                        # Model doesn't exist, create it
                        try:
                            self.client.create_registered_model(model_name)
                            print(f"   ✓ Created model '{model_name}' in registry")
                        except Exception as create_error:
                            if "already exists" in str(create_error).lower():
                                print(f"   ℹ️  Model '{model_name}' already exists")
                            else:
                                raise
                    
                    # Try to create a version using the client
                    try:
                        model_version = self.client.create_model_version(
                            name=model_name,
                            source=model_uri,
                            run_id=run_id
                        )
                        print(f"Successfully registered model '{model_name}'.")
                        print(f"   ✓ Version: {model_version.version}")
                        print(f"   ✓ Run ID: {run_id}")
                        return model_version
                    except Exception as version_error:
                        print(f"   ⚠️  Could not create version: {str(version_error)}")
                        # Check if version was created anyway
                        try:
                            latest_version = self.get_latest_version(model_name)
                            if latest_version:
                                print(f"   ✓ Found existing version: {latest_version.version}")
                                return latest_version
                        except:
                            pass
                        
                        print(f"   ❌ Model registration failed due to DagHub limitations")
                        print(f"   ℹ️  Note: DagHub has limited Model Registry support")
                        print(f"   ℹ️  The model may need to be registered manually via the DagsHub UI")
                        return None
                        
                except Exception as alt_error:
                    print(f"   ❌ Alternative registration method also failed: {str(alt_error)}")
                    return None
            else:
                # For other errors, show the actual error
                print(f"   ⚠️  Registration error: {error_str}")
                # Model might already be registered, get latest version
                try:
                    latest_version = self.get_latest_version(model_name)
                    if latest_version:
                        print(f"   ℹ️  Found existing version: {latest_version.version}")
                        return latest_version
                except:
                    pass
                return None
    
    def get_latest_version(self, model_name):
        """Get latest version of a registered model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                print(f"   ℹ️  Latest version: {latest.version}")
                return latest
            return None
        except Exception as e:
            print(f"   ⚠️  Could not get version: {str(e)}")
            return None
    
    def transition_model_stage(self, model_name, version, stage):
        """
        Transition model to a stage
        Stages: None, Staging, Production, Archived
        """
        print(f"\n🔄 Transitioning model to '{stage}'...")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production")
            )
            
            print(f"   ✓ {model_name} v{version} → {stage}")
            
        except Exception as e:
            print(f"   ❌ Transition failed: {str(e)}")
    
    def add_model_description(self, model_name, version, description):
        """Add description to a model version"""
        try:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            print(f"   ✓ Description added to {model_name} v{version}")
        except Exception as e:
            print(f"   ⚠️  Could not add description: {str(e)}")
    
    def list_registered_models(self):
        """List all registered models"""
        models = self.client.search_registered_models()
        
        if not models:
            print("\n📦 No registered models found")
            return []
        
        print("\n📦 Registered Models:")
        for model in models:
            print(f"\n   Model: {model.name}")
            
            # Get versions
            try:
                versions = self.client.search_model_versions(f"name='{model.name}'")
                if versions:
                    for v in versions:
                        print(f"      - Version {v.version}: {v.current_stage} (Run ID: {v.run_id})")
                else:
                    print(f"      ⚠️  No versions registered for this model")
                    print(f"      ℹ️  This may be due to DagHub API limitations")
            except Exception as e:
                print(f"      ⚠️  Could not retrieve versions: {str(e)}")
        
        return models
    
    def get_production_model(self, model_name):
        """Get the production version of a model"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            
            if versions:
                prod_version = versions[0]
                print(f"\n🚀 Production Model: {model_name}")
                print(f"   Version: {prod_version.version}")
                print(f"   Run ID: {prod_version.run_id}")
                return prod_version
            else:
                print(f"\n⚠️  No production version for {model_name}")
                return None
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None


def register_best_model(
    experiment_name="crypto-price-prediction",
    model_name="crypto-price-predictor",
    metric="val_rmse",
    auto_promote_to_staging=True
):
    """
    Complete workflow: Find best model and register it
    """
    print("\n" + "="*70)
    print("MODEL REGISTRY WORKFLOW")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Model Name: {model_name}")
    print(f"Selection Metric: {metric}")
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Get best run
    best_run = registry.get_best_run(experiment_name, metric=metric, ascending=True)
    
    # Register model
    model_version = registry.register_model(
        run_id=best_run.info.run_id,
        model_name=model_name
    )
    
    if model_version:
        # Add description
        description = f"""
        Model: {best_run.data.params.get('model_type', 'Unknown')}
        Validation RMSE: {best_run.data.metrics.get('val_rmse', 'N/A')}
        Validation R²: {best_run.data.metrics.get('val_r2', 'N/A')}
        Registered: {datetime.now().isoformat()}
        Run ID: {best_run.info.run_id}
        """
        registry.add_model_description(model_name, model_version.version, description)
        
        # Auto-promote to staging
        if auto_promote_to_staging:
            registry.transition_model_stage(
                model_name,
                model_version.version,
                "Staging"
            )
    
    # List all registered models
    registry.list_registered_models()
    
    print("\n" + "="*70)
    print("✅ MODEL REGISTRATION COMPLETE")
    print("="*70 + "\n")
    
    return model_version


def promote_to_production(model_name="crypto-price-predictor", version=None):
    """Promote a model version to production"""
    print("\n" + "="*70)
    print("PROMOTE MODEL TO PRODUCTION")
    print("="*70)
    
    registry = ModelRegistry()
    
    if version is None:
        # Get latest staging version
        try:
            versions = registry.client.get_latest_versions(model_name, stages=["Staging"])
            if versions:
                version = versions[0].version
                print(f"   Using latest Staging version: {version}")
            else:
                print("   ❌ No Staging version found")
                return None
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None
    
    # Promote to production
    registry.transition_model_stage(model_name, version, "Production")
    
    print("\n" + "="*70)
    print("✅ MODEL PROMOTED TO PRODUCTION")
    print("="*70 + "\n")
    
    return version


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "register":
            # Register best model
            experiment = sys.argv[2] if len(sys.argv) > 2 else "crypto-price-prediction"
            model_name = sys.argv[3] if len(sys.argv) > 3 else "crypto-price-predictor"
            register_best_model(experiment, model_name)
        
        elif command == "promote":
            # Promote to production
            model_name = sys.argv[2] if len(sys.argv) > 2 else "crypto-price-predictor"
            version = sys.argv[3] if len(sys.argv) > 3 else None
            promote_to_production(model_name, version)
        
        elif command == "list":
            # List all models
            registry = ModelRegistry()
            registry.list_registered_models()
        
        else:
            print("Unknown command. Use: register, promote, or list")
    
    else:
        print("""
Usage:
  python register_model.py register [experiment_name] [model_name]
  python register_model.py promote [model_name] [version]
  python register_model.py list
        """)