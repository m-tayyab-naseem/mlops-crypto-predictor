"""
Compare Models Across Multiple Runs
Generate comprehensive comparison reports
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class ModelComparator:
    """Compare multiple model runs"""
    
    def __init__(self, experiment_name="crypto-price-prediction"):
        """Initialize comparator"""
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        
        # Get experiment
        self.experiment = self.client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        print(f"✅ Experiment: {experiment_name}")
        print(f"   ID: {self.experiment.experiment_id}")
    
    def get_all_runs(self, max_results=50):
        """Get all runs from experiment"""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        print(f"\n📊 Found {len(runs)} runs")
        
        return runs
    
    def create_comparison_dataframe(self, runs):
        """Create DataFrame from runs for comparison"""
        data = []
        
        for run in runs:
            row = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'status': run.info.status,
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                row[f'param_{key}'] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                row[f'metric_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        print(f"\n📋 Comparison DataFrame:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        return df
    
    def get_top_models(self, df, metric='metric_val_rmse', n=5, ascending=True):
        """Get top N models by metric"""
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not found")
            return pd.DataFrame()
        
        top_df = df.nsmallest(n, metric) if ascending else df.nlargest(n, metric)
        
        print(f"\n🏆 Top {n} Models by {metric}:")
        for idx, row in top_df.iterrows():
            print(f"   {idx+1}. {row.get('param_model_type', 'Unknown')}: {row[metric]:.6f}")
        
        return top_df
    
    def plot_metric_comparison(self, df, metrics=['metric_val_rmse', 'metric_val_r2'], output_dir='reports'):
        """Plot metric comparisons across runs"""
        print(f"\n📈 Generating metric comparison plots...")
        
        # Filter to valid metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            print("   ⚠️  No metrics found for plotting")
            return None
        
        # Create figure
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            # Group by model type
            if 'param_model_type' in df.columns:
                df_plot = df.groupby('param_model_type')[metric].apply(list).to_dict()
                
                # Box plot
                data_to_plot = [values for values in df_plot.values()]
                labels = list(df_plot.keys())
                
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(f'{metric.replace("metric_", "").replace("_", " ").title()}')
                ax.set_xlabel('Model Type')
                ax.set_ylabel('Value')
                ax.grid(axis='y', alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # Simple scatter
                ax.scatter(range(len(df)), df[metric], alpha=0.6)
                ax.set_title(f'{metric.replace("metric_", "").replace("_", " ").title()}')
                ax.set_xlabel('Run Index')
                ax.set_ylabel('Value')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'metric_comparison_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {output_file}")
        
        return output_file
    
    def plot_model_performance_trend(self, df, metric='metric_val_rmse', output_dir='reports'):
        """Plot performance trend over time"""
        print(f"\n📊 Generating performance trend plot...")
        
        if metric not in df.columns or 'start_time' not in df.columns:
            print("   ⚠️  Required columns not found")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by time
        df_sorted = df.sort_values('start_time')
        
        # Plot by model type
        if 'param_model_type' in df.columns:
            for model_type in df_sorted['param_model_type'].unique():
                mask = df_sorted['param_model_type'] == model_type
                ax.plot(
                    df_sorted[mask]['start_time'],
                    df_sorted[mask][metric],
                    marker='o',
                    label=model_type,
                    alpha=0.7
                )
        else:
            ax.plot(
                df_sorted['start_time'],
                df_sorted[metric],
                marker='o',
                alpha=0.7
            )
        
        ax.set_xlabel('Training Time')
        ax.set_ylabel(metric.replace('metric_', '').replace('_', ' ').title())
        ax.set_title('Model Performance Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'performance_trend_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {output_file}")
        
        return output_file
    
    def generate_report(self, output_dir='reports'):
        """Generate comprehensive comparison report"""
        print("\n" + "="*70)
        print("MODEL COMPARISON REPORT")
        print("="*70)
        
        # Get all runs
        runs = self.get_all_runs()
        
        if not runs:
            print("⚠️  No runs found")
            return None
        
        # Create comparison DataFrame
        df = self.create_comparison_dataframe(runs)
        
        # Get top models
        top_models = self.get_top_models(df, metric='metric_val_rmse', n=5)
        
        # Generate plots
        plot1 = self.plot_metric_comparison(
            df,
            metrics=['metric_val_rmse', 'metric_val_r2', 'metric_test_rmse']
        )
        
        plot2 = self.plot_model_performance_trend(df, metric='metric_val_rmse')
        
        # Save DataFrame
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(output_dir, f'runs_comparison_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Saved comparison data: {csv_file}")
        
        # Summary statistics
        print(f"\n📊 Summary Statistics:")
        if 'metric_val_rmse' in df.columns:
            print(f"   Val RMSE - Mean: {df['metric_val_rmse'].mean():.6f}")
            print(f"   Val RMSE - Std:  {df['metric_val_rmse'].std():.6f}")
            print(f"   Val RMSE - Min:  {df['metric_val_rmse'].min():.6f}")
            print(f"   Val RMSE - Max:  {df['metric_val_rmse'].max():.6f}")
        
        print("\n" + "="*70)
        print("✅ REPORT GENERATED")
        print("="*70 + "\n")
        
        return {
            'dataframe': df,
            'top_models': top_models,
            'plots': [plot1, plot2],
            'csv_file': csv_file
        }


if __name__ == "__main__":
    import sys
    
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "crypto-price-prediction"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    
    comparator = ModelComparator(experiment_name)
    report = comparator.generate_report(output_dir)
    
    if report:
        print(f"\n📁 Report saved to: {output_dir}/")