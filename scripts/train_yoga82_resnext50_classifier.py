"""
ResNeXt50 Average Performance Evaluation (Unified Script for 6/20/82-Class)

This script trains the ResNeXt50 model 5 times with different random seeds
to compute average performance and standard deviation. Automatically detects
the number of classes from the input CSV file.

Features:
- Runs 5 independent training sessions
- Computes mean ± std for all metrics
- Automatically detects dataset type (6/20/82-class)
- Saves detailed results for each run
- Creates comparison plots across runs
- Generates comprehensive summary report

Usage:
    python average_train_resnext50_classifier.py --data output_dataset/yoga_poses_dataset_vit_6class.csv
    python average_train_resnext50_classifier.py --data output_dataset/yoga_poses_dataset_vit_20class.csv
    python average_train_resnext50_classifier.py --data output_dataset/yoga_poses_dataset_vit_82vanilla.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import time
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ResNeXtBlock(nn.Module):
    """ResNeXt block with grouped convolutions."""
    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super(ResNeXtBlock, self).__init__()
        
        group_width = out_channels // cardinality
        
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Linear(out_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class ResNeXt50KeypointClassifier(nn.Module):
    """ResNeXt50-inspired network for keypoint-based pose classification."""
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(ResNeXt50KeypointClassifier, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ResNeXt blocks
        self.block1 = ResNeXtBlock(256, 512)
        self.block2 = ResNeXtBlock(512, 512)
        self.block3 = ResNeXtBlock(512, 1024)
        self.block4 = ResNeXtBlock(1024, 1024)
        self.block5 = ResNeXtBlock(1024, 2048)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.classifier(x)
        
        return x

class YogaPoseCNNClassifier:
    """ResNeXt50-based yoga pose classifier."""
    
    def __init__(self, random_state=42, device=None):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.class_mapping = None
        self.feature_names = None
        self.metrics = {}
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
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
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
        """Create PyTorch DataLoaders."""
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, num_classes, input_dim, 
                   epochs=200, lr=0.001, patience=15, verbose=False):
        """Train the ResNeXt50 model."""
        # Initialize model
        self.model = ResNeXt50KeypointClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=0.5
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Print progress every 10 epochs if verbose
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        train_time = time.time() - start_time
        
        self.metrics['train_time'] = train_time
        self.metrics['best_val_acc'] = best_val_acc
        self.metrics['epochs_trained'] = epoch + 1
        self.metrics['train_history'] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        
        return train_losses, val_losses, train_accs, val_accs
    
    def evaluate(self, test_loader):
        """Comprehensive evaluation of the model."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        
        # Calculate metrics
        test_acc = accuracy_score(y_true, y_pred)
        top5_acc = top_k_accuracy_score(y_true, y_probs, k=5)
        
        # Precision, Recall, F1
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
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

def run_single_experiment(run_id, data_path, device, verbose=False):
    """Run a single training experiment with a specific random seed."""
    random_state = 42 + run_id  # Different seed for each run
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RUN {run_id + 1}/5 - Random Seed: {random_state}")
        print(f"{'='*80}")
    
    # Initialize classifier
    classifier = YogaPoseCNNClassifier(random_state=random_state, device=device)
    
    # Load data
    X, y = classifier.load_data(data_path)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(X, y)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm = classifier.normalize_features(
        X_train, X_val, X_test
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = classifier.create_dataloaders(
        X_train_norm, y_train,
        X_val_norm, y_val,
        X_test_norm, y_test,
        batch_size=64
    )
    
    # Train model
    num_classes = len(np.unique(y))
    input_dim = X_train_norm.shape[1]
    
    classifier.train_model(
        train_loader, val_loader,
        num_classes=num_classes,
        input_dim=input_dim,
        epochs=200,
        lr=0.001,
        patience=15,
        verbose=verbose
    )
    
    # Evaluate model
    metrics = classifier.evaluate(test_loader)
    
    if verbose:
        print(f"\n  Results for Run {run_id + 1}:")
        print(f"    Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"    Top-5 Accuracy: {metrics['test_top5_accuracy']:.4f}")
        print(f"    Precision (macro): {metrics['test_precision_macro']:.4f}")
        print(f"    Recall (macro): {metrics['test_recall_macro']:.4f}")
        print(f"    F1-Score (macro): {metrics['test_f1_macro']:.4f}")
        print(f"    Training time: {metrics['train_time']/60:.2f} minutes")
    
    return metrics

def compute_statistics(all_metrics):
    """Compute mean and standard deviation for all metrics."""
    metrics_summary = {}
    
    metric_keys = [
        'test_accuracy', 'test_top5_accuracy',
        'test_precision_micro', 'test_precision_macro', 'test_precision_weighted',
        'test_recall_micro', 'test_recall_macro', 'test_recall_weighted',
        'test_f1_micro', 'test_f1_macro', 'test_f1_weighted',
        'train_time', 'best_val_acc', 'epochs_trained'
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
    
    time_mean = metrics_summary['train_time']['mean'] / 60
    time_std = metrics_summary['train_time']['std'] / 60
    summary_text.append(f"Training time (minutes):    {time_mean:.2f} ± {time_std:.2f}")
    
    epochs_mean = metrics_summary['epochs_trained']['mean']
    epochs_std = metrics_summary['epochs_trained']['std']
    summary_text.append(f"Epochs trained:             {epochs_mean:.1f} ± {epochs_std:.1f}")
    
    val_acc_mean = metrics_summary['best_val_acc']['mean'] * 100
    val_acc_std = metrics_summary['best_val_acc']['std'] * 100
    summary_text.append(f"Best validation accuracy:   {val_acc_mean:.2f} ± {val_acc_std:.2f}%")
    
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
               for k, v in metrics.items() if k != 'train_history'}
        }
        runs_data.append(run_data)
    
    with open(save_dir / 'resnext50_individual_runs.json', 'w') as f:
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
    
    with open(save_dir / 'resnext50_average_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n  Results saved to:")
    print(f"    {save_dir / 'resnext50_individual_runs.json'}")
    print(f"    {save_dir / 'resnext50_average_results.json'}")

def save_summary_text(summary_text, save_dir):
    """Save summary text to a .txt file."""
    save_dir = Path(save_dir)
    summary_file = save_dir / 'resnext50_5_run_summary.txt'
    
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
        ('train_time', 'Training Time (min)')
    ]
    
    for idx, (metric_key, title) in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = metrics_summary[metric_key]['values']
        if metric_key == 'train_time':
            values = [v/60 for v in values]  # Convert to minutes
        elif metric_key != 'train_time':
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
    plt.savefig(save_dir / 'resnext50_runs_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    {save_dir / 'resnext50_runs_comparison.png'}")
    plt.close()

def detect_dataset_type(csv_path):
    """Detect dataset type (6/20/82-class) from CSV file."""
    df = pd.read_csv(csv_path)
    num_classes = len(df['class_id'].unique())
    
    if num_classes <= 6:
        return "6class"
    elif num_classes <= 20:
        return "20class"
    else:
        return "82class"

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description='ResNeXt50 Yoga Pose Classifier')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to CSV dataset (e.g., output_dataset/yoga_poses_dataset_vit_6class.csv)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs (default: 200)')
    
    args = parser.parse_args()
    
    # Detect dataset type
    dataset_type = detect_dataset_type(args.data)
    num_classes = pd.read_csv(args.data)['class_id'].nunique()
    
    print("\n" + "=" * 80)
    print(f"RESNEXT50 AVERAGE PERFORMANCE EVALUATION ({dataset_type.upper()})")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 5 independent training sessions...")
    
    overall_start = time.time()
    
    # Configuration
    data_path = args.data
    results_dir = Path(f"results/resnext50_{dataset_type}_5run_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    all_metrics = []
    for run_id in range(5):
        metrics = run_single_experiment(run_id, data_path, device, verbose=True)
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
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
