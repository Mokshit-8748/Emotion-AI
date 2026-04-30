import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
from src.gan_integration import compare_with_without_gan, prepare_gan_augmented_dataset
from src.data_preparation import prepare_data
from src.train_model import train_emotion_model
from src.evaluate_model import evaluate_model
from src.model import TemporalAttention

def detailed_model_comparison(data_path):
    """
    Comprehensive comparison between original and GAN-enhanced models
    """
    print("🔍 DETAILED MODEL COMPARISON")
    print("=" * 60)
    
    # Load original data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data_path)
    
    # Train original model
    print("\n📊 Training Original Model (No GAN)...")
    original_model = train_emotion_model(X_train, y_train)
    original_results = evaluate_model(original_model, X_test, y_test)
    
    # Prepare GAN-augmented data
    print("\n🤖 Preparing GAN-Augmented Data...")
    X_train_gan, X_test_gan, y_train_gan, y_test_gan = prepare_gan_augmented_dataset(
        data_path, use_gan=True, augmentation_factor=0.5
    )
    
    # Train GAN-enhanced model
    print("\n📈 Training GAN-Enhanced Model...")
    gan_model = train_emotion_model(X_train_gan, y_train_gan)
    gan_results = evaluate_model(gan_model, X_test_gan, y_test_gan)
    
    # Detailed comparison
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    # Overall accuracy comparison
    print(f"\n🎯 Overall Accuracy:")
    print(f"Original Model:  {original_results['accuracy']:.4f} ({original_results['accuracy']*100:.2f}%)")
    print(f"GAN Model:       {gan_results['accuracy']:.4f} ({gan_results['accuracy']*100:.2f}%)")
    improvement = (gan_results['accuracy'] - original_results['accuracy']) * 100
    print(f"Improvement:     {improvement:+.2f}%")
    
    # Class-wise comparison
    print(f"\n📋 Class-wise Performance:")
    print("-" * 40)
    emotions = label_encoder.classes_
    
    comparison_data = []
    for emotion in emotions:
        orig_precision = original_results['class_report'][emotion]['precision']
        gan_precision = gan_results['class_report'][emotion]['precision']
        orig_recall = original_results['class_report'][emotion]['recall']
        gan_recall = gan_results['class_report'][emotion]['recall']
        orig_f1 = original_results['class_report'][emotion]['f1-score']
        gan_f1 = gan_results['class_report'][emotion]['f1-score']
        
        print(f"\n{emotion.upper()}:")
        print(f"  Precision: {orig_precision:.3f} → {gan_precision:.3f} ({(gan_precision-orig_precision)*100:+.1f}%)")
        print(f"  Recall:    {orig_recall:.3f} → {gan_recall:.3f} ({(gan_recall-orig_recall)*100:+.1f}%)")
        print(f"  F1-Score:  {orig_f1:.3f} → {gan_f1:.3f} ({(gan_f1-orig_f1)*100:+.1f}%)")
        
        comparison_data.append({
            'Emotion': emotion,
            'Original_Precision': orig_precision,
            'GAN_Precision': gan_precision,
            'Original_Recall': orig_recall,
            'GAN_Recall': gan_recall,
            'Original_F1': orig_f1,
            'GAN_F1': gan_f1
        })
    
    # Data size comparison
    print(f"\n📏 Dataset Size Comparison:")
    print(f"Original Training Set: {len(X_train)} samples")
    # Scaler is now MinMaxScaler
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'data_min_'):
        print(f"Data Min range: {np.min(scaler.data_min_):.4f} to {np.max(scaler.data_min_):.4f}")
        print(f"Data Max range: {np.min(scaler.data_max_):.4f} to {np.max(scaler.data_max_):.4f}")
    else:
        print("Scaler is not MinMaxScaler or not fitted.")
    print(f"GAN Training Set:      {len(X_train_gan)} samples")
    print(f"Augmentation Factor:   {len(X_train_gan)/len(X_train):.2f}x")
    
    # Visual comparison
    create_comparison_plots(original_results, gan_results, emotions, comparison_data)
    
    return {
        'original': original_results,
        'gan': gan_results,
        'data_sizes': {
            'original_train': len(X_train),
            'gan_train': len(X_train_gan),
            'augmentation_factor': len(X_train_gan)/len(X_train)
        }
    }

def create_comparison_plots(original_results, gan_results, emotions, comparison_data):
    """
    Create visual comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Original vs GAN-Enhanced Model Comparison', fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(comparison_data)
    
    # Precision comparison
    axes[0, 0].bar(df['Emotion'], df['Original_Precision'], alpha=0.7, label='Original', color='skyblue')
    axes[0, 0].bar(df['Emotion'], df['GAN_Precision'], alpha=0.7, label='GAN', color='lightcoral')
    axes[0, 0].set_title('Precision Comparison')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    axes[0, 1].bar(df['Emotion'], df['Original_Recall'], alpha=0.7, label='Original', color='skyblue')
    axes[0, 1].bar(df['Emotion'], df['GAN_Recall'], alpha=0.7, label='GAN', color='lightcoral')
    axes[0, 1].set_title('Recall Comparison')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    axes[1, 0].bar(df['Emotion'], df['Original_F1'], alpha=0.7, label='Original', color='skyblue')
    axes[1, 0].bar(df['Emotion'], df['GAN_F1'], alpha=0.7, label='GAN', color='lightcoral')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Overall accuracy comparison
    models = ['Original', 'GAN-Enhanced']
    accuracies = [original_results['accuracy'], gan_results['accuracy']]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[1, 1].bar(models, accuracies, color=colors)
    axes[1, 1].set_title('Overall Accuracy Comparison')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f} ({acc*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def confusion_matrix_comparison(data_path):
    """
    Compare confusion matrices side by side
    """
    print("\n🔍 CONFUSION MATRIX COMPARISON")
    print("=" * 40)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data_path)
    
    # Train models
    original_model = train_emotion_model(X_train, y_train)
    
    # Get GAN-augmented data
    X_train_gan, X_test_gan, y_train_gan, y_test_gan = prepare_gan_augmented_dataset(
        data_path, use_gan=True, augmentation_factor=0.5
    )
    gan_model = train_emotion_model(X_train_gan, y_train_gan)
    
    # Get predictions
    y_pred_original = original_model(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), training=False).numpy()
    y_pred_original = np.argmax(y_pred_original, axis=1)
    
    y_pred_gan = gan_model(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), training=False).numpy()
    y_pred_gan = np.argmax(y_pred_gan, axis=1)
    
    # Create confusion matrices
    cm_original = confusion_matrix(y_test, y_pred_original)
    cm_gan = confusion_matrix(y_test, y_pred_gan)
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    
    emotions = label_encoder.classes_
    
    # Original model confusion matrix
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions, ax=axes[0])
    axes[0].set_title('Original Model')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # GAN model confusion matrix
    sns.heatmap(cm_gan, annot=True, fmt='d', cmap='Reds', 
                xticklabels=emotions, yticklabels=emotions, ax=axes[1])
    axes[1].set_title('GAN-Enhanced Model')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm_original, cm_gan

def training_history_comparison(data_path):
    """
    Compare training histories if available
    """
    print("\n📈 TRAINING HISTORY COMPARISON")
    print("=" * 40)
    
    # This would require modifying the training function to return history
    # For now, we'll show data size impact
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data_path)
    X_train_gan, _, y_train_gan, _ = prepare_gan_augmented_dataset(
        data_path, use_gan=True, augmentation_factor=0.5
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Data Comparison', fontsize=16, fontweight='bold')
    
    # Class distribution comparison
    unique_orig, counts_orig = np.unique(y_train, return_counts=True)
    unique_gan, counts_gan = np.unique(y_train_gan, return_counts=True)
    
    emotions = label_encoder.classes_[unique_orig]
    
    ax1.bar(emotions, counts_orig, alpha=0.7, label='Original', color='skyblue')
    ax1.bar(emotions, counts_gan[:len(unique_orig)], alpha=0.7, label='GAN-Augmented', color='lightcoral')
    ax1.set_title('Training Data Distribution')
    ax1.set_ylabel('Number of Samples')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Augmentation factor by class
    augmentation_factors = counts_gan[:len(unique_orig)] / counts_orig
    ax2.bar(emotions, augmentation_factors, color='green', alpha=0.7)
    ax2.set_title('Augmentation Factor by Class')
    ax2.set_ylabel('Augmentation Factor (x)')
    ax2.axhline(y=1.5, color='red', linestyle='--', label='Target (1.5x)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('training_data_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data_path = "data/RAVDESS"
    
    print("🚀 Starting Comprehensive Model Comparison...")
    print("This will compare your original model with the GAN-enhanced version.")
    print("Please ensure you have enough time as this may take a while...")
    
    # Run comprehensive comparison
    results = detailed_model_comparison(data_path)
    
    # Confusion matrix comparison
    cm_orig, cm_gan = confusion_matrix_comparison(data_path)
    
    # Training data comparison
    training_history_comparison(data_path)
    
    print("\n✅ Comparison Complete!")
    print("Generated visualizations:")
    print("- model_comparison.png")
    print("- confusion_matrix_comparison.png") 
    print("- training_data_comparison.png")
    
    print(f"\n📊 Key Results:")
    print(f"Original Accuracy: {results['original']['accuracy']:.4f}")
    print(f"GAN Accuracy: {results['gan']['accuracy']:.4f}")
    print(f"Improvement: {(results['gan']['accuracy'] - results['original']['accuracy'])*100:+.2f}%")
    print(f"Data Augmentation: {results['data_sizes']['augmentation_factor']:.2f}x")
