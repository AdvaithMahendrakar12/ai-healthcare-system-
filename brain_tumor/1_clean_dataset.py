
import os
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import json

class DatasetCleaner:
    def __init__(self):
        self.raw_data_dir = 'brain_tumor/datasets/brain_tumor'
        self.cleaned_data_dir = 'brain_tumor/cleaned_data'
        self.target_size = (224, 224)
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'corrupted_images': 0,
            'cleaned_images': 0,
            'class_distribution': {}
        }
    
    def create_directories(self):
        """Create cleaned data directory structure"""
        print("📁 Creating directory structure...")
        
        for split in ['Training', 'Testing']:
            for class_name in self.classes:
                path = os.path.join(self.cleaned_data_dir, split, class_name)
                os.makedirs(path, exist_ok=True)
        
        os.makedirs('brain_tumor/models', exist_ok=True)
        os.makedirs('brain_tumor/results', exist_ok=True)
        print("✅ Directories created!")
    
    def is_valid_image(self, img_path):
        """Check if image is valid and not corrupted"""
        try:
            img = Image.open(img_path)
            img.verify()  # Verify it's actually an image
            img = Image.open(img_path)  # Reopen after verify
            img = np.array(img)
            
            # Check if image has proper dimensions
            if img.shape[0] < 50 or img.shape[1] < 50:
                return False
            
            return True
        except Exception as e:
            return False
    
    def clean_and_resize_image(self, img_path, output_path):
        """Clean, resize, and normalize image"""
        try:
            # Read image
            img = cv2.imread(img_path)
            
            if img is None:
                return False
            
            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Enhance contrast (CLAHE)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Save cleaned image
            cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return False
    
    def clean_dataset(self):
        """Main cleaning function"""
        print("\n" + "="*70)
        print("🧹 STARTING DATASET CLEANING")
        print("="*70)
        
        for split in ['Training', 'Testing']:
            print(f"\n📊 Processing {split} data...")
            split_path = os.path.join(self.raw_data_dir, split)
            
            if not os.path.exists(split_path):
                print(f"⚠️ Warning: {split_path} not found. Skipping...")
                continue
            
            for class_name in self.classes:
                class_path = os.path.join(split_path, class_name)
                output_class_path = os.path.join(self.cleaned_data_dir, split, class_name)
                
                if not os.path.exists(class_path):
                    print(f"⚠️ Warning: {class_path} not found. Skipping...")
                    continue
                
                # Get all images
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"\n  Processing {class_name}: {len(image_files)} images")
                
                valid_count = 0
                corrupted_count = 0
                
                for img_file in tqdm(image_files, desc=f"  {class_name}"):
                    img_path = os.path.join(class_path, img_file)
                    output_path = os.path.join(output_class_path, img_file)
                    
                    self.stats['total_images'] += 1
                    
                    # Validate image
                    if not self.is_valid_image(img_path):
                        corrupted_count += 1
                        self.stats['corrupted_images'] += 1
                        continue
                    
                    # Clean and save
                    if self.clean_and_resize_image(img_path, output_path):
                        valid_count += 1
                        self.stats['cleaned_images'] += 1
                    else:
                        corrupted_count += 1
                        self.stats['corrupted_images'] += 1
                
                # Update class distribution
                key = f"{split}_{class_name}"
                self.stats['class_distribution'][key] = valid_count
                
                print(f"  ✅ Valid: {valid_count}, ❌ Corrupted: {corrupted_count}")
    
    def generate_report(self):
        """Generate cleaning report"""
        print("\n" + "="*70)
        print("📊 CLEANING REPORT")
        print("="*70)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Successfully cleaned: {self.stats['cleaned_images']}")
        print(f"Corrupted/Invalid: {self.stats['corrupted_images']}")
        print(f"Success rate: {(self.stats['cleaned_images']/self.stats['total_images']*100):.2f}%")
        
        print("\n📈 Class Distribution:")
        for key, count in sorted(self.stats['class_distribution'].items()):
            print(f"  {key}: {count} images")
        
        # Save report
        with open('brain_tumor/results/cleaning_report.json', 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        print("\n✅ Report saved to: brain_tumor/results/cleaning_report.json")
        print("="*70)
    
    def run(self):
        """Run complete cleaning pipeline"""
        self.create_directories()
        self.clean_dataset()
        self.generate_report()
        
        print("\n🎉 Dataset cleaning complete!")
        print("📂 Cleaned data saved to:", self.cleaned_data_dir)
        print("➡️  Next step: Run python 2_train_model.py")

if __name__ == "__main__":
    cleaner = DatasetCleaner()
    cleaner.run()