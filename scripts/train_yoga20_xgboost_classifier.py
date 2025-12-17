"""
Phase 3: XGBoost Average Performance Evaluation (20-Class Dataset)

This script trains the XGBoost classifier 5 times with different random seeds
to compute average performance and standard deviation on the 20-class dataset.

Features:
- Runs 5 independent training sessions
- Computes mean ± std for all metrics
- Saves detailed results for each run
- Creates comparison plots across runs
- Generates comprehensive summary report
- Saves results to xgboost_20class_5run_results/ folder
- Creates text files with summary statistics and model architecture info
- Saves best performing model (monitored by validation accuracy) for each run
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class YogaPoseClassifier:
    """XGBoost-based yoga pose classifier."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.class_mapping = None
        self.feature_names = None
        self.metrics = {}
        
    def load_data(self, csv_path):
        """Load and prepare data from CSV."""
        df = pd.read_csv(csv_path)
        
        # Load class mapping
        mapping_path = csv_path.replace('.csv', '_class_mapping.json')
        if Path(mapping_path).exists():
            with open(mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
                self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        
        # Extract features: only x, y coordinates
        feature_cols = [col for col in df.columns 
                       if (col.endswith('_x') or col.endswith('_y')) 
                       and not col.endswith('_conf')]
        
        X = df[feature_cols].values
        y = df['class_id'].values
        
        self.feature_names = feature_cols
        
        # Handle missing values
        missing = np.isnan(X).sum()
        if missing > 0:
            X = np.nan_to_num(X, 0)
        
        return X, y
    
    def split_data(self, X, y):
        """Split data: 70% train, 15% val, 15% test."""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.random_state, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train, X_val, X_test):
        """Normalize features using StandardScaler."""
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def train_model(self, X_train, y_train, X_val, y_val, verbose=False, model_save_path=None):
        """Train XGBoost model with tuned hyperparameters."""
        start_time = time.time()
        
        # Use tuned hyperparameters (from previous tuning)
        params = {
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'seed': self.random_state,
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 0.8
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10 if verbose else 0
        )
        
        train_time = time.time() - start_time
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        y_val_pred_proba = self.model.predict_proba(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_top5_acc = top_k_accuracy_score(y_val, y_val_pred_proba, k=5)
        
        self.metrics['train_time'] = train_time
        self.metrics['best_val_acc'] = val_acc
        self.metrics['best_val_top5_acc'] = val_top5_acc
        self.metrics['epochs_trained'] = self.model.best_iteration if hasattr(self.model, 'best_iteration') else params['n_estimators']
        
        # Save best model to file if path provided
        if model_save_path is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'class_mapping': self.class_mapping,
                'feature_names': self.feature_names,
                'val_acc': val_acc,
                'val_top5_acc': val_top5_acc,
                'params': params
            }
            with open(model_save_path, 'wb') as f:
                pickle.dump(model_data, f)
        
        return val_acc
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation of the model."""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, y_pred)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
        
        # Precision, Recall, F1
        precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store metrics
        self.metrics.update({
            'test_accuracy': test_acc,
            'test_top5_accuracy': top5_acc,
            'test_precision_micro': precision_micro,
            'test_precision_macro': precision_macro,
            'test_precision_weighted': precision_weighted,
            'test_recall_micro': recall_micro,
            'test_recall_macro': recall_macro,
            'test_recall_weighted': recall_weighted,
            'test_f1_micro': f1_micro,
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted
        })
        
        return self.metrics

def run_single_experiment(run_id, data_path, save_dir, verbose=False):
    """Run a single training experiment with a specific random seed."""
    random_state = 42 + run_id  # Different seed for each run
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RUN {run_id + 1}/5 - Random Seed: {random_state}")
        print(f"{'='*80}")
    
    # Initialize classifier
    classifier = YogaPoseClassifier(random_state=random_state)
    
    # Load data
    X, y = classifier.load_data(data_path)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(X, y)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm = classifier.normalize_features(
        X_train, X_val, X_test
    )
    
    # Model save path for best model
    model_save_path = save_dir / f'best_model_run_{run_id + 1}.pkl'
    
    # Train model
    classifier.train_model(
        X_train_norm, y_train,
        X_val_norm, y_val,
        verbose=verbose,
        model_save_path=model_save_path
    )
    
    # Evaluate model
    metrics = classifier.evaluate(X_test_norm, y_test)
    
    if verbose:
        print(f"\n  Results for Run {run_id + 1}:")
        print(f"    Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"    Top-5 Accuracy: {metrics['test_top5_accuracy']:.4f}")
        print(f"    Precision (macro): {metrics['test_precision_macro']:.4f}")
        print(f"    Recall (macro): {metrics['test_recall_macro']:.4f}")
        print(f"    F1-Score (macro): {metrics['test_f1_macro']:.4f}")
        print(f"    Best Val Acc: {metrics['best_val_acc']:.4f}")
        print(f"    Training time: {metrics['train_time']:.2f} seconds")
        print(f"    Best model saved: {model_save_path}")
    
    return metrics

def compute_statistics(all_metrics):
    """Compute mean and standard deviation for all metrics."""
    metrics_summary = {}
    
    metric_keys = [
        'test_accuracy', 'test_top5_accuracy',
        'test_precision_micro', 'test_precision_macro', 'test_precision_weighted',
        'test_recall_micro', 'test_recall_macro', 'test_recall_weighted',
        'test_f1_micro', 'test_f1_macro', 'test_f1_weighted',
        'train_time', 'best_val_acc', 'best_val_top5_acc', 'epochs_trained'
    ]
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        metrics_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    return metrics_summary

def print_summary(metrics_summary):
    """Print formatted summary with mean ± std."""
    summary_text = []
    
    summary_text.append("\n" + "=" * 80)
    summary_text.append("AVERAGE PERFORMANCE ACROSS 5 RUNS")
    summary_text.append("=" * 80)
    
    summary_text.append("\n" + "-" * 80)
    summary_text.append("ACCURACY METRICS")
    summary_text.append("-" * 80)
    
    acc_mean = metrics_summary['test_accuracy']['mean'] * 100
    acc_std = metrics_summary['test_accuracy']['std'] * 100
    summary_text.append(f"Test Accuracy:              {acc_mean:.2f} ± {acc_std:.2f}%")
    
    top5_mean = metrics_summary['test_top5_accuracy']['mean'] * 100
    top5_std = metrics_summary['test_top5_accuracy']['std'] * 100
    summary_text.append(f"Top-5 Accuracy:             {top5_mean:.2f} ± {top5_std:.2f}%")
    
    summary_text.append("\n" + "-" * 80)
    summary_text.append("PRECISION METRICS")
    summary_text.append("-" * 80)
    
    prec_micro_mean = metrics_summary['test_precision_micro']['mean'] * 100
    prec_micro_std = metrics_summary['test_precision_micro']['std'] * 100
    summary_text.append(f"Precision (micro):          {prec_micro_mean:.2f} ± {prec_micro_std:.2f}%")
    
    prec_macro_mean = metrics_summary['test_precision_macro']['mean'] * 100
    prec_macro_std = metrics_summary['test_precision_macro']['std'] * 100
    summary_text.append(f"Precision (macro):          {prec_macro_mean:.2f} ± {prec_macro_std:.2f}%")
    
    prec_weighted_mean = metrics_summary['test_precision_weighted']['mean'] * 100
    prec_weighted_std = metrics_summary['test_precision_weighted']['std'] * 100
    summary_text.append(f"Precision (weighted):       {prec_weighted_mean:.2f} ± {prec_weighted_std:.2f}%")
    
    summary_text.append("\n" + "-" * 80)
    summary_text.append("RECALL METRICS")
    summary_text.append("-" * 80)
    
    recall_micro_mean = metrics_summary['test_recall_micro']['mean'] * 100
    recall_micro_std = metrics_summary['test_recall_micro']['std'] * 100
    summary_text.append(f"Recall (micro):             {recall_micro_mean:.2f} ± {recall_micro_std:.2f}%")
    
    recall_macro_mean = metrics_summary['test_recall_macro']['mean'] * 100
    recall_macro_std = metrics_summary['test_recall_macro']['std'] * 100
    summary_text.append(f"Recall (macro):             {recall_macro_mean:.2f} ± {recall_macro_std:.2f}%")
    
    recall_weighted_mean = metrics_summary['test_recall_weighted']['mean'] * 100
    recall_weighted_std = metrics_summary['test_recall_weighted']['std'] * 100
    summary_text.append(f"Recall (weighted):          {recall_weighted_mean:.2f} ± {recall_weighted_std:.2f}%")
    
    summary_text.append("\n" + "-" * 80)
    summary_text.append("F1-SCORE METRICS")
    summary_text.append("-" * 80)
    
    f1_micro_mean = metrics_summary['test_f1_micro']['mean'] * 100
    f1_micro_std = metrics_summary['test_f1_micro']['std'] * 100
    summary_text.append(f"F1-Score (micro):           {f1_micro_mean:.2f} ± {f1_micro_std:.2f}%")
    
    f1_macro_mean = metrics_summary['test_f1_macro']['mean'] * 100
    f1_macro_std = metrics_summary['test_f1_macro']['std'] * 100
    summary_text.append(f"F1-Score (macro):           {f1_macro_mean:.2f} ± {f1_macro_std:.2f}%")
    
    f1_weighted_mean = metrics_summary['test_f1_weighted']['mean'] * 100
    f1_weighted_std = metrics_summary['test_f1_weighted']['std'] * 100
    summary_text.append(f"F1-Score (weighted):        {f1_weighted_mean:.2f} ± {f1_weighted_std:.2f}%")
    
    summary_text.append("\n" + "-" * 80)
    summary_text.append("TRAINING STATISTICS")
    summary_text.append("-" * 80)
    
    time_mean = metrics_summary['train_time']['mean']
    time_std = metrics_summary['train_time']['std']
    summary_text.append(f"Training time (seconds):    {time_mean:.2f} ± {time_std:.2f}")
    
    epochs_mean = metrics_summary['epochs_trained']['mean']
    epochs_std = metrics_summary['epochs_trained']['std']
    summary_text.append(f"Epochs trained:             {epochs_mean:.1f} ± {epochs_std:.1f}")
    
    val_acc_mean = metrics_summary['best_val_acc']['mean'] * 100
    val_acc_std = metrics_summary['best_val_acc']['std'] * 100
    summary_text.append(f"Best validation accuracy:   {val_acc_mean:.2f} ± {val_acc_std:.2f}%")
    
    val_top5_mean = metrics_summary['best_val_top5_acc']['mean'] * 100
    val_top5_std = metrics_summary['best_val_top5_acc']['std'] * 100
    summary_text.append(f"Best val top-5 accuracy:    {val_top5_mean:.2f} ± {val_top5_std:.2f}%")
    
    # Print to console
    for line in summary_text:
        print(line)
    
    return summary_text

def save_results(all_metrics, metrics_summary, save_dir):
    """Save all results to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Save individual run results
    runs_data = []
    for i, metrics in enumerate(all_metrics):
        run_data = {
            'run_id': i + 1,
            'random_seed': 42 + i,
            **{k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
               for k, v in metrics.items()}
        }
        runs_data.append(run_data)
    
    with open(save_dir / 'xgboost_individual_runs.json', 'w') as f:
        json.dump(runs_data, f, indent=2)
    
    # Save summary statistics
    summary_data = {}
    for key, stats in metrics_summary.items():
        summary_data[key] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max'])
        }
    
    with open(save_dir / 'xgboost_average_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n  Results saved to:")
    print(f"    {save_dir / 'xgboost_individual_runs.json'}")
    print(f"    {save_dir / 'xgboost_average_results.json'}")

def save_summary_text(summary_text, save_dir):
    """Save summary text to a .txt file."""
    save_dir = Path(save_dir)
    summary_file = save_dir / 'xgboost_5_run_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_text))
    
    print(f"    {summary_file}")

def plot_results(all_metrics, metrics_summary, save_dir):
    """Create visualization of results across runs."""
    save_dir = Path(save_dir)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_to_plot = [
        ('test_accuracy', 'Test Accuracy'),
        ('test_top5_accuracy', 'Top-5 Accuracy'),
        ('test_precision_macro', 'Precision (Macro)'),
        ('test_recall_macro', 'Recall (Macro)'),
        ('test_f1_macro', 'F1-Score (Macro)'),
        ('train_time', 'Training Time (sec)')
    ]
    
    for idx, (metric_key, title) in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = metrics_summary[metric_key]['values']
        if metric_key != 'train_time':
            values = [v*100 for v in values]  # Convert to percentage
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Bar plot
        x = np.arange(1, 6)
        bars = ax.bar(x, values, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mean line
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        # Shaded std region
        ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='red', label=f'Std: {std_val:.2f}')
        
        ax.set_xlabel('Run', fontsize=12)
        ylabel = title if metric_key == 'train_time' else f'{title} (%)'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'xgboost_runs_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    {save_dir / 'xgboost_runs_comparison.png'}")
    plt.close()

def save_model_summary(results_dir, num_features, num_classes):
    """Save model architecture summary to text file."""
    summary_file = results_dir / 'xgboost_model_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("XGBOOST MODEL ARCHITECTURE SUMMARY (20-CLASS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Model Type: XGBoost Classifier\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Input Features: {num_features}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("Objective: multi:softmax\n")
        f.write("Tree Method: hist\n")
        f.write("Max Depth: 8\n")
        f.write("Learning Rate: 0.1\n")
        f.write("Number of Estimators: 300\n")
        f.write("Min Child Weight: 1\n")
        f.write("Subsample: 1.0\n")
        f.write("Colsample by Tree: 0.8\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MODEL CHARACTERISTICS\n")
        f.write("-" * 80 + "\n")
        f.write("Model Type: Gradient Boosting Decision Trees\n")
        f.write("Ensemble Method: Boosting\n")
        f.write("Base Learners: Decision Trees\n")
        f.write("Training Strategy: Iterative boosting with early stopping\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PARAMETER INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write("Note: XGBoost models don't have a fixed parameter count.\n")
        f.write("The number of parameters grows with:\n")
        f.write("  - Number of trees (n_estimators)\n")
        f.write("  - Tree depth (max_depth)\n")
        f.write("  - Number of features and classes\n\n")
        
        f.write("Approximate model size depends on:\n")
        f.write(f"  - Trees: {300}\n")
        f.write(f"  - Max depth: {8}\n")
        f.write(f"  - Features: {num_features}\n")
        f.write(f"  - Classes: {num_classes}\n\n")
        
    print(f"    {summary_file}")

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 80)
    print("XGBOOST AVERAGE PERFORMANCE EVALUATION (20-CLASS)")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 5 independent training sessions...")
    
    overall_start = time.time()
    
    # Configuration
    data_path = "output_dataset/yoga_poses_dataset_vit_20class.csv"
    results_dir = Path("results/xgboost_20class_5run_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data to get dimensions for model summary
    import pandas as pd
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns 
                   if (col.endswith('_x') or col.endswith('_y')) 
                   and not col.endswith('_conf')]
    num_features = len(feature_cols)
    num_classes = len(df['class_id'].unique())
    
    # Run experiments
    all_metrics = []
    for run_id in range(5):
        metrics = run_single_experiment(run_id, data_path, results_dir, verbose=True)
        all_metrics.append(metrics)
        
        # Print intermediate summary after each run
        print(f"\n  Run {run_id + 1} completed:")
        print(f"    Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"    F1 (macro): {metrics['test_f1_macro']*100:.2f}%")
    
    # Compute statistics
    print("\n" + "=" * 80)
    print("COMPUTING STATISTICS...")
    print("=" * 80)
    metrics_summary = compute_statistics(all_metrics)
    
    # Print summary and get text
    summary_text = print_summary(metrics_summary)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)
    save_results(all_metrics, metrics_summary, results_dir)
    save_summary_text(summary_text, results_dir)
    save_model_summary(results_dir, num_features, num_classes)
    
    # Plot results
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS...")
    print("=" * 80)
    plot_results(all_metrics, metrics_summary, results_dir)
    
    overall_time = time.time() - overall_start
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"\nTotal time: {overall_time/60:.2f} minutes")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    acc_mean = metrics_summary['test_accuracy']['mean'] * 100
    acc_std = metrics_summary['test_accuracy']['std'] * 100
    print(f"\nTest Accuracy: {acc_mean:.2f} ± {acc_std:.2f}%")
    
    f1_mean = metrics_summary['test_f1_macro']['mean'] * 100
    f1_std = metrics_summary['test_f1_macro']['std'] * 100
    print(f"F1-Score (macro): {f1_mean:.2f} ± {f1_std:.2f}%")
    
    print(f"\nBest models saved:")
    for i in range(5):
        print(f"  Run {i+1}: {results_dir / f'best_model_run_{i+1}.pkl'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
