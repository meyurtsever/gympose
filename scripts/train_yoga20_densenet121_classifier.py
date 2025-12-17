"""
Phase 3: DenseNet121 Average Performance Evaluation (20-Class Dataset)

This script trains the DenseNet121 model 5 times with different random seeds
to compute average performance and standard deviation on the 20-class dataset.

Features:
- Runs 5 independent training sessions
- Computes mean ± std for all metrics
- Saves detailed results for each run
- Creates comparison plots across runs
- Generates comprehensive summary report
- Saves results to densenet121_20class_5run_results/ folder
- Creates text files with summary statistics and model architecture info
- Saves best performing model (monitored by validation loss) for each run
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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DenseBlock(nn.Module):
    """Dense Block with multiple dense layers."""
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(self._make_dense_layer(layer_in_channels, growth_rate, dropout))
    
    def _make_dense_layer(self, in_channels, growth_rate, dropout):
        """Create a single dense layer (BN-ReLU-Linear)."""
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, growth_rate),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    """Transition layer between dense blocks."""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.transition(x)

class DenseNet121KeypointClassifier(nn.Module):
    """DenseNet121 architecture adapted for 1D keypoint data."""
    def __init__(self, input_dim, num_classes, growth_rate=32, dropout=0.2):
        super(DenseNet121KeypointClassifier, self).__init__()
        
        # DenseNet-121 configuration: [6, 12, 24, 16] layers per block
        num_layers_per_block = [6, 12, 24, 16]
        
        # Initial convolution
        num_init_features = 64
        self.features = nn.Sequential(
            nn.Linear(input_dim, num_init_features),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        # Dense Blocks and Transition Layers
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(num_layers_per_block):
            # Dense Block
            block = DenseBlock(
                in_channels=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers,
                dropout=dropout
            )
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            
            # Transition Layer (except after last block)
            if i != len(num_layers_per_block) - 1:
                trans = TransitionLayer(
                    in_channels=num_features,
                    out_channels=num_features // 2,
                    dropout=dropout
                )
                self.transitions.append(trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.final_bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.final_bn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class YogaPoseDenseNetClassifier:
    """DenseNet121 based yoga pose classifier."""
    
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
                   epochs=100, lr=0.001, patience=20, warmup_epochs=10, 
                   verbose=False, model_save_path=None):
        """Train the DenseNet121 model."""
        # Initialize model
        self.model = DenseNet121KeypointClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            growth_rate=32,
            dropout=0.2
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        
        # Learning rate scheduler with warmup
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
            
            # Learning rate scheduling with warmup
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_loss)
            
            # Print progress every 10 epochs if verbose
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict()
                
                # Save best model to file if path provided
                if model_save_path is not None:
                    model_data = {
                        'model_state_dict': best_model_state,
                        'scaler': self.scaler,
                        'class_mapping': self.class_mapping,
                        'feature_names': self.feature_names,
                        'epoch': epoch + 1,
                        'val_loss': best_val_loss,
                        'val_acc': best_val_acc,
                        'model_config': {
                            'input_dim': input_dim,
                            'num_classes': num_classes
                        }
                    }
                    torch.save(model_data, model_save_path)
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
        self.metrics['best_val_loss'] = best_val_loss
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

def run_single_experiment(run_id, data_path, device, save_dir, verbose=False):
    """Run a single training experiment with a specific random seed."""
    random_state = 42 + run_id  # Different seed for each run
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RUN {run_id + 1}/5 - Random Seed: {random_state}")
        print(f"{'='*80}")
    
    # Initialize classifier
    classifier = YogaPoseDenseNetClassifier(random_state=random_state, device=device)
    
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
    
    # Model save path for best model
    model_save_path = save_dir / f'best_model_run_{run_id + 1}.pth'
    
    classifier.train_model(
        train_loader, val_loader,
        num_classes=num_classes,
        input_dim=input_dim,
        epochs=100,
        lr=0.001,
        patience=20,
        warmup_epochs=10,
        verbose=verbose,
        model_save_path=model_save_path
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
        print(f"    Best Val Loss: {metrics['best_val_loss']:.4f}")
        print(f"    Training time: {metrics['train_time']/60:.2f} minutes")
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
        'train_time', 'best_val_loss', 'best_val_acc', 'epochs_trained'
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
    
    val_loss_mean = metrics_summary['best_val_loss']['mean']
    val_loss_std = metrics_summary['best_val_loss']['std']
    summary_text.append(f"Best validation loss:       {val_loss_mean:.4f} ± {val_loss_std:.4f}")
    
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
    
    with open(save_dir / 'densenet121_individual_runs.json', 'w') as f:
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
    
    with open(save_dir / 'densenet121_average_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n  Results saved to:")
    print(f"    {save_dir / 'densenet121_individual_runs.json'}")
    print(f"    {save_dir / 'densenet121_average_results.json'}")

def save_summary_text(summary_text, save_dir):
    """Save summary text to a .txt file."""
    save_dir = Path(save_dir)
    summary_file = save_dir / 'densenet121_5_run_summary.txt'
    
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
    plt.savefig(save_dir / 'densenet121_runs_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    {save_dir / 'densenet121_runs_comparison.png'}")
    plt.close()

def save_model_summary(results_dir, model, num_features, num_classes):
    """Save model architecture summary to text file."""
    summary_file = results_dir / 'densenet121_model_summary.txt'
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DENSENET121 MODEL ARCHITECTURE SUMMARY (20-CLASS)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Model Type: DenseNet121 Keypoint Classifier\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Input Features: {num_features}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Parameter Size: {total_params * 4 / (1024**2):.2f} MB (float32)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("ARCHITECTURE (DenseNet121 Configuration)\n")
        f.write("-" * 80 + "\n")
        f.write("Initial Features:\n")
        f.write(f"  - Linear: {num_features} -> 64\n")
        f.write("  - BatchNorm1d(64) + ReLU\n\n")
        
        f.write("Dense Block 1:\n")
        f.write("  - 6 dense layers (growth_rate=32)\n")
        f.write("  - Each layer: BN -> ReLU -> Linear -> Dropout(0.2)\n")
        f.write("  - Output channels: 64 + 6*32 = 256\n\n")
        
        f.write("Transition Layer 1:\n")
        f.write("  - BN -> ReLU -> Linear (256 -> 128)\n\n")
        
        f.write("Dense Block 2:\n")
        f.write("  - 12 dense layers (growth_rate=32)\n")
        f.write("  - Output channels: 128 + 12*32 = 512\n\n")
        
        f.write("Transition Layer 2:\n")
        f.write("  - BN -> ReLU -> Linear (512 -> 256)\n\n")
        
        f.write("Dense Block 3:\n")
        f.write("  - 24 dense layers (growth_rate=32)\n")
        f.write("  - Output channels: 256 + 24*32 = 1024\n\n")
        
        f.write("Transition Layer 3:\n")
        f.write("  - BN -> ReLU -> Linear (1024 -> 512)\n\n")
        
        f.write("Dense Block 4:\n")
        f.write("  - 16 dense layers (growth_rate=32)\n")
        f.write("  - Output channels: 512 + 16*32 = 1024\n\n")
        
        f.write("Final Layers:\n")
        f.write("  - BatchNorm1d(1024) + ReLU\n")
        f.write("  - Dropout(0.2)\n")
        f.write(f"  - Linear: 1024 -> {num_classes}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("KEY FEATURES\n")
        f.write("-" * 80 + "\n")
        f.write("- Dense Connectivity: Each layer receives features from all preceding layers\n")
        f.write("- Growth Rate: 32 features added per layer\n")
        f.write("- Total Dense Layers: 6 + 12 + 24 + 16 = 58 layers\n")
        f.write("- Transition Layers: Reduce feature maps by 50% between blocks\n")
        f.write("- Efficient Parameter Usage: Feature reuse reduces redundancy\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write("Optimizer: AdamW (lr=0.001, weight_decay=0.05)\n")
        f.write("Scheduler: LinearLR warmup (10 epochs) + ReduceLROnPlateau\n")
        f.write("Loss Function: CrossEntropyLoss\n")
        f.write("Gradient Clipping: max_norm=1.0\n")
        f.write("Epochs: 100 (with early stopping, patience=20)\n")
        f.write("Batch Size: 64\n\n")
        
    print(f"    {summary_file}")

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 80)
    print("DENSENET121 AVERAGE PERFORMANCE EVALUATION (20-CLASS)")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning 5 independent training sessions...")
    
    overall_start = time.time()
    
    # Configuration
    data_path = "output_dataset/yoga_poses_dataset_vit_20class.csv"
    results_dir = Path("results/densenet121_20class_5run_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data to get dimensions for model summary
    import pandas as pd
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns 
                   if (col.endswith('_x') or col.endswith('_y')) 
                   and not col.endswith('_conf')]
    num_features = len(feature_cols)
    num_classes = len(df['class_id'].unique())
    
    # Create a sample model for summary
    sample_model = DenseNet121KeypointClassifier(
        input_dim=num_features,
        num_classes=num_classes,
        growth_rate=32,
        dropout=0.2
    ).to(device)
    
    # Run experiments
    all_metrics = []
    for run_id in range(5):
        metrics = run_single_experiment(run_id, data_path, device, results_dir, verbose=True)
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
    save_model_summary(results_dir, sample_model, num_features, num_classes)
    
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
        print(f"  Run {i+1}: {results_dir / f'best_model_run_{i+1}.pth'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
