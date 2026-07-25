"""
Test Lung X-ray Model Accuracy - Complete Version
Run this AFTER starting model_server.py
"""

import os
from pathlib import Path
from modules.model_service import model_service

def find_images_in_folder(folder_path):
    """Recursively find all images in a folder and subfolders"""
    image_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(Path(root) / file)
    
    return image_files

def test_model_accuracy():
    """Test model accuracy on all images"""
    print("=" * 60)
    print("🔬 TESTING LUNG X-RAY MODEL ACCURACY")
    print("=" * 60)
    
    # First check if model server is running
    status = model_service.get_status()
    if not status['loaded']:
        print("❌ Model server not ready!")
        print("   Please run: python model_server.py first")
        return
    
    print(f"✅ Model server connected")
    print()
    
    # Base path
    base_path = Path('datasets/lung_xray')
    
    # Categories to test
    categories = [
        ('Normal', base_path / 'normal', 'Normal'),
        ('Pneumonia', base_path / 'pneumonia', 'Pneumonia'),
        ('COVID', base_path / 'covid', 'COVID')
    ]
    
    all_results = {}
    
    for display_name, folder_path, expected in categories:
        print(f"\n📂 Testing {display_name} images...")
        print(f"   Path: {folder_path}")
        
        # Find all images recursively
        image_files = find_images_in_folder(folder_path)
        
        print(f"   Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"   ⚠️ No images found in {display_name} folder")
            continue
        
        # Test up to 100 images per category (or all if less)
        total = len(image_files)  # Test ALL images (no min limit)
        correct = 0
        results_details = []
        
        print(f"   Testing {total} images...")
        
        for i, img_path in enumerate(image_files[:total]):
            try:
                result = model_service.predict(str(img_path))
                
                if result.get('success'):
                    predicted = result.get('prediction', '')
                    confidence = result.get('confidence', 0)
                    
                    # Check if prediction matches (case-insensitive)
                    is_correct = predicted.lower() == expected.lower()
                    
                    if is_correct:
                        correct += 1
                        mark = "✅"
                    else:
                        mark = "❌"
                    
                    results_details.append({
                        'file': img_path.name,
                        'expected': expected,
                        'predicted': predicted,
                        'correct': is_correct,
                        'confidence': confidence
                    })
                    
                    # Show progress every 20 images
                    if (i + 1) % 500 == 0 or i == total - 1:
                        print(f"      Progress: {i+1}/{total} - Current accuracy: {correct/(i+1)*100:.1f}%")
                        
                else:
                    print(f"      ❌ Error on {img_path.name}: {result.get('error')}")
                    
            except Exception as e:
                print(f"      ⚠️ Error testing {img_path.name}: {e}")
        
        # Calculate accuracy
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        all_results[display_name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'details': results_details
        }
        
        print(f"\n   📊 {display_name} Results:")
        print(f"      Correct: {correct}/{total}")
        print(f"      Accuracy: {accuracy:.1f}%")
        
        # Show some examples
        if results_details:
            print(f"\n      Sample predictions:")
            # Show first 3 correct and first 3 wrong
            correct_samples = [d for d in results_details if d['correct']][:3]
            wrong_samples = [d for d in results_details if not d['correct']][:3]
            
            if correct_samples:
                print(f"      ✅ Correct examples:")
                for s in correct_samples:
                    print(f"         {s['file']}: {s['predicted']} ({s['confidence']:.1f}%)")
            
            if wrong_samples:
                print(f"      ❌ Wrong examples:")
                for s in wrong_samples:
                    print(f"         {s['file']}: got {s['predicted']}, expected {s['expected']} ({s['confidence']:.1f}%)")
    
    # Overall results
    print("\n" + "=" * 60)
    print("🏆 FINAL ACCURACY RESULTS")
    print("=" * 60)
    
    total_correct = 0
    total_tested = 0
    
    for category, data in all_results.items():
        if data['total'] > 0:
            print(f"\n{category}:")
            print(f"   Correct: {data['correct']}/{data['total']}")
            print(f"   Accuracy: {data['accuracy']:.1f}%")
            total_correct += data['correct']
            total_tested += data['total']
    
    if total_tested > 0:
        overall = (total_correct / total_tested) * 100
        print(f"\n" + "=" * 60)
        print(f"🎯 OVERALL ACCURACY: {overall:.1f}%")
        print(f"   Tested on {total_tested} images total")
        print("=" * 60)
        
        # Calculate per-class stats
        if len(all_results) == 3:
            print(f"\n📊 Summary by Class:")
            for category, data in all_results.items():
                if data['total'] > 0:
                    print(f"   {category:10}: {data['accuracy']:5.1f}% ({data['correct']:3d}/{data['total']:3d})")
    else:
        print("\n❌ No images tested! Please check your dataset folders.")

if __name__ == "__main__":
    test_model_accuracy()