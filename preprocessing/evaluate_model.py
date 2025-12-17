"""
Evaluate trained XGBoost model with additional metrics (Precision, Recall, etc.)
Loads the saved model and computes comprehensive evaluation metrics without retraining.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_model(model_path):
    """Load the trained model and associated components."""
    print("=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\nModel loaded from: {model_path}")
    print(f"Model type: {type(model_data['model']).__name__}")
    print(f"Number of features: {len(model_data['feature_names'])}")
    if model_data['class_mapping']:
        print(f"Number of classes: {len(set(model_data['class_mapping'].keys()))}")
    else:
        print(f"Number of classes: {model_data['model'].n_classes_}")
    
    return model_data

def load_and_prepare_data(csv_path, model_data):
    """Load data and prepare it the same way as training."""
    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    
    # Extract features (same as training)
    feature_cols = [col for col in df.columns 
                   if (col.endswith('_x') or col.endswith('_y')) 
                   and not col.endswith('_conf')]
    
    X = df[feature_cols].values
    y = df['class_id'].values
    
    # Handle missing values
    X = np.nan_to_num(X, 0)
    
    # Split data (same random state as training)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Normalize using the saved scaler
    scaler = model_data['scaler']
    X_test_norm = scaler.transform(X_test)
    X_val_norm = scaler.transform(X_val)
    
    print(f"Test set: {len(X_test)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    return X_test_norm, y_test, X_val_norm, y_val

def calculate_comprehensive_metrics(model, X_test, y_test, class_mapping):
    """Calculate comprehensive evaluation metrics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall metrics
    metrics = {}
    
    # Accuracy
    metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
    metrics['test_top5_accuracy'] = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    
    # Precision (micro, macro, weighted)
    metrics['test_precision_micro'] = precision_score(y_test, y_pred, average='micro', zero_division=0)
    metrics['test_precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['test_precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Recall (micro, macro, weighted)
    metrics['test_recall_micro'] = recall_score(y_test, y_pred, average='micro', zero_division=0)
    metrics['test_recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['test_recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # F1-Score (micro, macro, weighted)
    metrics['test_f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
    metrics['test_f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['test_f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    
    print(f"\nAccuracy:")
    print(f"  Overall Accuracy:        {metrics['test_accuracy']:.4f}")
    print(f"  Top-5 Accuracy:          {metrics['test_top5_accuracy']:.4f}")
    
    print(f"\nPrecision (how many predicted positives are actually positive):")
    print(f"  Micro-averaged:          {metrics['test_precision_micro']:.4f}")
    print(f"  Macro-averaged:          {metrics['test_precision_macro']:.4f}")
    print(f"  Weighted-averaged:       {metrics['test_precision_weighted']:.4f}")
    
    print(f"\nRecall (how many actual positives are correctly identified):")
    print(f"  Micro-averaged:          {metrics['test_recall_micro']:.4f}")
    print(f"  Macro-averaged:          {metrics['test_recall_macro']:.4f}")
    print(f"  Weighted-averaged:       {metrics['test_recall_weighted']:.4f}")
    
    print(f"\nF1-Score (harmonic mean of precision and recall):")
    print(f"  Micro-averaged:          {metrics['test_f1_micro']:.4f}")
    print(f"  Macro-averaged:          {metrics['test_f1_macro']:.4f}")
    print(f"  Weighted-averaged:       {metrics['test_f1_weighted']:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Extract per-class metrics
    class_metrics = []
    for class_id in range(len(np.unique(y_test))):
        class_id_str = str(class_id)
        if class_id_str in report:
            precision = report[class_id_str]['precision']
            recall = report[class_id_str]['recall']
            f1 = report[class_id_str]['f1-score']
            support = report[class_id_str]['support']
            class_name = class_mapping.get(class_id, f"Class_{class_id}") if class_mapping else f"Class_{class_id}"
            class_metrics.append({
                'class_id': class_id,
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(support)
            })
    
    # Sort by F1-score
    class_metrics.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print("\nTop 15 Classes (Best Performance):")
    print(f"{'#':<4} {'Class Name':<45} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 95)
    for i, cm in enumerate(class_metrics[:15], 1):
        print(f"{i:<4} {cm['class_name'][:44]:<45} {cm['precision']:<10.4f} {cm['recall']:<10.4f} {cm['f1_score']:<10.4f} {cm['support']:<8}")
    
    print("\nBottom 15 Classes (Worst Performance):")
    print(f"{'#':<4} {'Class Name':<45} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 95)
    for i, cm in enumerate(class_metrics[-15:], 1):
        print(f"{i:<4} {cm['class_name'][:44]:<45} {cm['precision']:<10.4f} {cm['recall']:<10.4f} {cm['f1_score']:<10.4f} {cm['support']:<8}")
    
    # Classes with perfect scores
    perfect_classes = [cm for cm in class_metrics if cm['f1_score'] == 1.0]
    if perfect_classes:
        print(f"\n{len(perfect_classes)} Classes with Perfect F1-Score (1.0):")
        for cm in perfect_classes:
            print(f"  - {cm['class_name'][:60]} (n={cm['support']})")
    
    # Classes with zero scores
    zero_classes = [cm for cm in class_metrics if cm['f1_score'] == 0.0]
    if zero_classes:
        print(f"\n{len(zero_classes)} Classes with Zero F1-Score (0.0):")
        for cm in zero_classes:
            print(f"  - {cm['class_name'][:60]} (n={cm['support']})")
    
    return metrics, class_metrics, report

def save_detailed_report(metrics, class_metrics, output_path):
    """Save detailed evaluation report."""
    report = {
        'overall_metrics': metrics,
        'per_class_metrics': class_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nDetailed report saved to: {output_path}")

def create_metrics_comparison_plot(metrics, save_path):
    """Create visualization comparing different metric averages."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    metric_types = ['Precision', 'Recall', 'F1-Score']
    micro = [metrics['test_precision_micro'], metrics['test_recall_micro'], metrics['test_f1_micro']]
    macro = [metrics['test_precision_macro'], metrics['test_recall_macro'], metrics['test_f1_macro']]
    weighted = [metrics['test_precision_weighted'], metrics['test_recall_weighted'], metrics['test_f1_weighted']]
    
    x = np.arange(len(metric_types))
    width = 0.25
    
    plt.bar(x - width, micro, width, label='Micro-averaged', alpha=0.8)
    plt.bar(x, macro, width, label='Macro-averaged', alpha=0.8)
    plt.bar(x + width, weighted, width, label='Weighted-averaged', alpha=0.8)
    
    plt.xlabel('Metric Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance: Precision, Recall, and F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, metric_types)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (m, ma, w) in enumerate(zip(micro, macro, weighted)):
        plt.text(i - width, m + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, ma + 0.02, f'{ma:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width, w + 0.02, f'{w:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison plot saved to: {save_path}")
    plt.close()

def main():
    print("\n" + "=" * 80)
    print("XGBOOST MODEL - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Paths
    model_path = "models/yoga_xgboost_classifier.pkl"
    data_path = "output_dataset/yoga_poses_dataset_vit.csv"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load model
    model_data = load_model(model_path)
    
    # Load and prepare data
    X_test, y_test, X_val, y_val = load_and_prepare_data(data_path, model_data)
    
    # Calculate comprehensive metrics
    metrics, class_metrics, report = calculate_comprehensive_metrics(
        model_data['model'],
        X_test,
        y_test,
        model_data['class_mapping']
    )
    
    # Save detailed report
    save_detailed_report(
        metrics,
        class_metrics,
        results_dir / 'xgboost_detailed_evaluation.json'
    )
    
    # Create visualization
    create_metrics_comparison_plot(
        metrics,
        results_dir / 'xgboost_metrics_comparison.png'
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)
    print("\nFiles created:")
    print(f"  - results/xgboost_detailed_evaluation.json")
    print(f"  - results/xgboost_metrics_comparison.png")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
