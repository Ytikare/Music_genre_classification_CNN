import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

plt.style.use('default')
sns.set_palette("husl")

class CNNTrainingLogVisualizer:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.training_data = {}
        self.per_class_metrics = {}
        self.system_info = {}
        
    def parse_log_file(self):
        with open(self.log_file_path, 'r') as f:
            content = f.read()
        
        epoch_pattern = r'Epoch (\d+)/50\n\d+/\d+ \[=+\] - \d+s \d+ms/step - loss: ([\d.]+) - accuracy: ([\d.]+) - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)'
        matches = re.findall(epoch_pattern, content)
        
        epochs = []
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        for match in matches:
            epochs.append(int(match[0]))
            train_loss.append(float(match[1]))
            train_acc.append(float(match[2]))
            val_loss.append(float(match[3]))
            val_acc.append(float(match[4]))
        
        self.training_data = {
            'epochs': epochs,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        
        class_pattern = r'- (\w+): Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)'
        class_matches = re.findall(class_pattern, content)
        
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for match in class_matches:
            classes.append(match[0])
            precision.append(float(match[1]))
            recall.append(float(match[2]))
            f1_score.append(float(match[3]))
        
        self.per_class_metrics = {
            'classes': classes,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        self.system_info = {
            'total_samples': 999,
            'training_samples': 699,
            'validation_samples': 150,
            'test_samples': 150,
            'final_test_accuracy': 0.9267,
            'training_time_hours': 5.45,  # 5 hours 27 minutes
            'model_parameters': 458122,
            'model_size_mb': 3.7,
            'avg_prediction_time': 0.164
        }
    
    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        ax1.plot(self.training_data['epochs'], self.training_data['train_loss'], 
                label='Training Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(self.training_data['epochs'], self.training_data['val_loss'], 
                label='Validation Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 50)
        
        # Accuracy curves
        ax2.plot(self.training_data['epochs'], 
                [acc * 100 for acc in self.training_data['train_accuracy']], 
                label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        ax2.plot(self.training_data['epochs'], 
                [acc * 100 for acc in self.training_data['val_accuracy']], 
                label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 50)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        classes = self.per_class_metrics['classes']
        precision = [p * 100 for p in self.per_class_metrics['precision']]
        recall = [r * 100 for r in self.per_class_metrics['recall']]
        f1_score = [f * 100 for f in self.per_class_metrics['f1_score']]
        
        bars1 = ax1.bar(classes, precision, color=sns.color_palette("viridis", len(classes)))
        ax1.set_title('Precision by Music Genre', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precision (%)')
        ax1.set_ylim(80, 100)
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(precision):
            ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        bars2 = ax2.bar(classes, recall, color=sns.color_palette("plasma", len(classes)))
        ax2.set_title('Recall by Music Genre', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall (%)')
        ax2.set_ylim(80, 100)
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(recall):
            ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        bars3 = ax3.bar(classes, f1_score, color=sns.color_palette("crest", len(classes)))
        ax3.set_title('F1-Score by Music Genre', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score (%)')
        ax3.set_ylim(80, 100)
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(f1_score):
            ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax4.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax4.bar(x, recall, width, label='Recall', alpha=0.8)
        ax4.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax4.set_xlabel('Music Genre')
        ax4.set_ylabel('Score (%)')
        ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(classes, rotation=45)
        ax4.legend()
        ax4.set_ylim(80, 100)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_overview(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        labels = ['Training', 'Validation', 'Test']
        sizes = [self.system_info['training_samples'], 
                self.system_info['validation_samples'], 
                self.system_info['test_samples']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
        
        time_per_epoch = self.system_info['training_time_hours'] * 60 / 50  # minutes per epoch
        cumulative_time = [i * time_per_epoch for i in range(1, 51)]
        
        ax2.plot(cumulative_time, [acc * 100 for acc in self.training_data['val_accuracy']], 
                linewidth=3, color='purple', marker='o', markersize=2)
        ax2.set_xlabel('Training Time (minutes)')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.set_title('Learning Progress Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        metrics_names = ['Test Accuracy', 'Best Val Accuracy', 'Model Size (MB)', 'Prediction Time (ms)']
        metrics_values = [self.system_info['final_test_accuracy'] * 100, 96.0, 
                         self.system_info['model_size_mb'], 
                         self.system_info['avg_prediction_time'] * 1000]
        
        bars = ax3.barh(metrics_names, metrics_values, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax3.set_title('Key Model Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Value')

        for i, (bar, value) in enumerate(zip(bars, metrics_values)):
            if i < 2:
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}%', va='center', fontweight='bold')
            elif i == 2:
                ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f} MB', va='center', fontweight='bold')
            else:
                ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                        f'{value:.0f} ms', va='center', fontweight='bold')
        
        genre_counts = [100, 100, 100, 100, 100, 99, 100, 100, 100, 100]  # jazz has 99
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        bars4 = ax4.bar(genres, genre_counts, color=sns.color_palette("Set3", len(genres)))
        ax4.set_title('Training Samples per Genre', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Samples')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(95, 102)
        
        bars4[5].set_color('red')
        bars4[5].set_alpha(0.7)
        
        for i, v in enumerate(genre_counts):
            ax4.text(i, v + 0.2, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix_heatmap(self):
        classes = self.per_class_metrics['classes']
        n_classes = len(classes)
        
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for i, (precision, recall) in enumerate(zip(self.per_class_metrics['precision'], 
                                                   self.per_class_metrics['recall'])):
            true_positives = int(recall * 15) 
            false_positives = int(true_positives / precision - true_positives)
            false_negatives = 15 - true_positives
            
            confusion_matrix[i, i] = true_positives
            
            if false_positives > 0:
                other_classes = list(range(n_classes))
                other_classes.remove(i)
                fp_distribution = np.random.multinomial(false_positives, 
                                                       [1/len(other_classes)] * len(other_classes))
                for j, fp in enumerate(fp_distribution):
                    confusion_matrix[other_classes[j], i] += fp
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Number of Samples'})
        plt.title('Simulated Confusion Matrix\n(Based on Precision/Recall Metrics)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Genre', fontsize=12)
        plt.ylabel('True Genre', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        print("="*60)
        print("CNN MUSIC GENRE CLASSIFICATION - TRAINING SUMMARY REPORT")
        print("="*60)
        print(f"Training Date: 2025-06-07")
        print(f"Total Training Time: {self.system_info['training_time_hours']:.2f} hours")
        print(f"Total Samples: {self.system_info['total_samples']}")
        print(f"Training/Validation/Test Split: {self.system_info['training_samples']}/{self.system_info['validation_samples']}/{self.system_info['test_samples']}")
        print()
        
        print("FINAL PERFORMANCE METRICS:")
        print(f"- Final Test Accuracy: {self.system_info['final_test_accuracy']*100:.2f}%")
        print(f"- Best Validation Accuracy: 96.00%")
        print(f"- Model Parameters: {self.system_info['model_parameters']:,}")
        print(f"- Model Size: {self.system_info['model_size_mb']} MB")
        print(f"- Average Prediction Time: {self.system_info['avg_prediction_time']*1000:.0f} ms")
        print()
        
        print("PER-CLASS PERFORMANCE:")
        for i, genre in enumerate(self.per_class_metrics['classes']):
            print(f"- {genre.capitalize():>10}: P={self.per_class_metrics['precision'][i]:.2f}, "
                  f"R={self.per_class_metrics['recall'][i]:.2f}, "
                  f"F1={self.per_class_metrics['f1_score'][i]:.2f}")
        print()
        
        # Best and worst performing genres
        f1_scores = self.per_class_metrics['f1_score']
        best_idx = np.argmax(f1_scores)
        worst_idx = np.argmin(f1_scores)
        
        print("INSIGHTS:")
        print(f"- Best performing genre: {self.per_class_metrics['classes'][best_idx]} (F1: {f1_scores[best_idx]:.2f})")
        print(f"- Most challenging genre: {self.per_class_metrics['classes'][worst_idx]} (F1: {f1_scores[worst_idx]:.2f})")
        print(f"- Training showed good convergence with minimal overfitting")
        print(f"- Model achieved stable performance after epoch 38")
        print("="*60)
    
    def create_all_visualizations(self):
        print("Parsing log file...")
        self.parse_log_file()
        
        print("Generating training curves...")
        self.plot_training_curves()
        
        print("Generating per-class metrics...")
        self.plot_per_class_metrics()
        
        print("Generating model overview...")
        self.plot_model_overview()
        
        print("Generating confusion matrix...")
        self.plot_confusion_matrix_heatmap()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print("\nAll visualizations saved as PNG files in the current directory!")

if __name__ == "__main__":
    visualizer = CNNTrainingLogVisualizer("training_process.log")
    
    visualizer.create_all_visualizations()