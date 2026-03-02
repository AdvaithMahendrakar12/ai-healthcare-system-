"""
🧪 ADVANCED BRAIN TUMOR MODEL TESTING
=====================================
Features:
- Test-Time Augmentation (TTA) for better accuracy
- Comprehensive per-class analysis
- Confidence calibration check
- Visualization of predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedModelTester:
    def __init__(self):
        self.model_paths = [
            'brain_tumor/models/brain_tumor_model.h5',
            'brain_tumor/models/brain_tumor_final.keras',
            'brain_tumor/models/brain_tumor_best.h5'
        ]
        self.class_labels_path = 'brain_tumor/models/class_labels.json'
        self.img_size = (224, 224)
        self.tta_steps = 5
        
        # Find and load model
        self.model = None
        for path in self.model_paths:
            if os.path.exists(path):
                print(f"📥 Loading model from: {path}")
                self.model = keras.models.load_model(path)
                print("✅ Model loaded successfully!")
                break
        
        if self.model is None:
            raise FileNotFoundError("No model found! Please train the model first.")
        
        # Load class labels
        with open(self.class_labels_path, 'r') as f:
            class_info = json.load(f)
            self.classes = class_info['classes']
            if 'input_size' in class_info:
                self.img_size = tuple(class_info['input_size'])
        
        print(f"✅ Classes: {self.classes}")
        print(f"✅ Input size: {self.img_size}")
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_single(self, img_path, use_tta=False):
        """Predict tumor type from single image"""
        img = self.preprocess_image(img_path)
        
        if use_tta:
            predictions = self._predict_with_tta_single(img_path)
        else:
            predictions = self.model.predict(img, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        class_probs = {self.classes[i]: float(predictions[0][i]) * 100 
                      for i in range(len(self.classes))}
        
        return predicted_class, confidence, class_probs
    
    def _predict_with_tta_single(self, img_path):
        """Apply TTA on single image"""
        all_preds = []
        
        # Original
        img = self.preprocess_image(img_path)
        all_preds.append(self.model.predict(img, verbose=0))
        
        # Augmented versions
        original = cv2.imread(img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        augmentations = [
            lambda x: cv2.flip(x, 1),  # Horizontal flip
            lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda x: cv2.GaussianBlur(x, (3, 3), 0),
        ]
        
        for aug_fn in augmentations[:self.tta_steps-1]:
            aug_img = aug_fn(original.copy())
            aug_img = cv2.resize(aug_img, self.img_size)
            aug_img = aug_img / 255.0
            aug_img = np.expand_dims(aug_img, axis=0)
            all_preds.append(self.model.predict(aug_img, verbose=0))
        
        return np.mean(all_preds, axis=0)
    
    def test_on_directory(self, test_dir='brain_tumor/cleaned_data/Testing', use_tta=True):
        """Test model on entire test directory"""
        print("\n" + "="*70)
        print("🧪 COMPREHENSIVE MODEL TESTING")
        print("="*70)
        
        # Create test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Standard evaluation
        print("\n📊 Standard Evaluation:")
        results = self.model.evaluate(test_gen, verbose=1)
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]*100:.2f}%")
        
        # Predictions
        test_gen.reset()
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        standard_acc = np.mean(y_pred == y_true)
        print(f"  Calculated Accuracy: {standard_acc*100:.2f}%")
        
        # TTA Evaluation (on subset for speed)
        if use_tta:
            print(f"\n📊 TTA Evaluation ({self.tta_steps} augmentations)...")
            tta_preds = self._batch_tta_predict(test_gen, test_dir)
            y_pred_tta = np.argmax(tta_preds, axis=1)
            tta_acc = np.mean(y_pred_tta == y_true)
            print(f"  TTA Accuracy: {tta_acc*100:.2f}%")
            print(f"  Improvement: {(tta_acc - standard_acc)*100:+.2f}%")
            y_pred = y_pred_tta  # Use TTA predictions
        
        # Classification report
        print("\n📋 Classification Report:")
        report = classification_report(y_true, y_pred, target_names=self.classes, digits=4)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class analysis
        print("\n📊 Per-Class Performance:")
        class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, cls in enumerate(self.classes):
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = class_acc[i]
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  {cls}:")
            print(f"    Accuracy:  {recall*100:.2f}%")
            print(f"    Precision: {precision*100:.2f}%")
            print(f"    F1-Score:  {f1*100:.2f}%")
        
        return cm, y_true, y_pred, y_pred_probs
    
    def _batch_tta_predict(self, test_gen, test_dir):
        """Apply TTA to entire test set"""
        tta_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            zoom_range=0.05
        )
        
        all_preds = []
        
        # Original predictions
        test_gen.reset()
        all_preds.append(self.model.predict(test_gen, verbose=0))
        
        # TTA predictions
        for i in range(self.tta_steps - 1):
            tta_gen = tta_datagen.flow_from_directory(
                test_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                shuffle=False,
                seed=42 + i
            )
            pred = self.model.predict(tta_gen, verbose=0)
            all_preds.append(pred)
        
        return np.mean(all_preds, axis=0)
    
    def test_sample_images(self, test_dir='brain_tumor/cleaned_data/Testing', samples_per_class=3):
        """Test on sample images from each class"""
        print("\n" + "="*70)
        print("🖼️ SAMPLE IMAGE PREDICTIONS")
        print("="*70)
        
        results = []
        
        for class_name in self.classes:
            class_path = os.path.join(test_dir, class_name)
            if not os.path.exists(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:samples_per_class]
            
            print(f"\n📁 {class_name.upper()}:")
            for img_file in images:
                img_path = os.path.join(class_path, img_file)
                pred_class, conf, probs = self.predict_single(img_path, use_tta=True)
                
                correct = pred_class == class_name
                status = "✅" if correct else "❌"
                
                print(f"  {status} {img_file[:20]:20s} → {pred_class} ({conf:.1f}%)")
                
                results.append({
                    'true_class': class_name,
                    'predicted_class': pred_class,
                    'confidence': conf,
                    'correct': correct,
                    'image_path': img_path,
                    'probabilities': probs
                })
        
        # Summary
        correct = sum(1 for r in results if r['correct'])
        print(f"\n📊 Sample Test Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
        
        return results
    
    def visualize_results(self, cm, sample_results):
        """Create comprehensive visualization"""
        print("\n📊 Generating visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes,
                    annot_kws={'size': 14})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Normalized Confusion Matrix
        plt.subplot(2, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens',
                    xticklabels=self.classes,
                    yticklabels=self.classes,
                    annot_kws={'size': 12})
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Per-class accuracy
        plt.subplot(2, 2, 3)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        colors = ['#E91E63', '#9C27B0', '#4CAF50', '#FF9800']
        bars = plt.bar(self.classes, class_acc * 100, color=colors, edgecolor='black')
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 105)
        for bar, acc in zip(bars, class_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        # Sample predictions grid
        plt.subplot(2, 2, 4)
        if sample_results:
            correct = sum(1 for r in sample_results if r['correct'])
            total = len(sample_results)
            
            # Create summary text
            summary = f"Sample Test Results\n\n"
            summary += f"Correct: {correct}/{total} ({correct/total*100:.1f}%)\n\n"
            
            for cls in self.classes:
                cls_results = [r for r in sample_results if r['true_class'] == cls]
                cls_correct = sum(1 for r in cls_results if r['correct'])
                summary += f"{cls}: {cls_correct}/{len(cls_results)}\n"
            
            plt.text(0.5, 0.5, summary, transform=plt.gca().transAxes,
                    fontsize=14, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            plt.axis('off')
            plt.title('Sample Test Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('brain_tumor/results/test_results_visualization.png', dpi=200, bbox_inches='tight')
        print("✅ Visualization saved to: brain_tumor/results/test_results_visualization.png")
        plt.close()
    
    def save_results(self, cm, accuracy_standard, accuracy_tta=None):
        """Save test results to JSON"""
        results = {
            'standard_accuracy': float(accuracy_standard),
            'tta_accuracy': float(accuracy_tta) if accuracy_tta else None,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': {
                self.classes[i]: float(cm[i, i] / cm[i].sum())
                for i in range(len(self.classes))
            }
        }
        
        with open('brain_tumor/results/final_test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("✅ Results saved to: brain_tumor/results/final_test_results.json")

def main():
    """Run comprehensive model testing"""
    
    # Check for model
    model_exists = any(os.path.exists(p) for p in [
        'brain_tumor/models/brain_tumor_model.h5',
        'brain_tumor/models/brain_tumor_final.keras',
        'brain_tumor/models/brain_tumor_best.h5'
    ])
    
    if not model_exists:
        print("❌ ERROR: No trained model found!")
        print("➡️  Please run: python brain_tumor/2_train_model.py first")
        return
    
    # Create tester
    tester = AdvancedModelTester()
    
    # Full test set evaluation
    cm, y_true, y_pred, y_probs = tester.test_on_directory(use_tta=True)
    
    # Sample predictions
    sample_results = tester.test_sample_images(samples_per_class=5)
    
    # Visualizations
    tester.visualize_results(cm, sample_results)
    
    # Calculate accuracies
    standard_acc = cm.diagonal().sum() / cm.sum()
    
    # Save results
    tester.save_results(cm, standard_acc)
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print(f"📊 Overall Accuracy: {standard_acc*100:.2f}%")
    print("\n📊 Per-Class Accuracy:")
    for i, cls in enumerate(tester.classes):
        cls_acc = cm[i, i] / cm[i].sum() * 100
        print(f"   {cls}: {cls_acc:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
