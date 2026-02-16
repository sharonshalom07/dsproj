"""
=============================================================================
Machine Learning-Assisted Skin Disease Classification System
Complete Pipeline: EDA → Preprocessing → Augmentation → ML-Ready Dataset
=============================================================================
"""

# ============================================================================
# STEP 0: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import hashlib
import warnings
import json
import pickle
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Scikit-image for feature extraction
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import exposure

# For handling imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Joblib for saving
import joblib

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS TO MATCH YOUR SETUP
# ============================================================================

CONFIG = {
    # === PATHS ===
    'dataset1_path': r'data\raw\dataset1',
    'dataset2_path': r'data\raw\dataset2',
    'combined_path': r'data\combined',
    'augmented_path': r'data\augmented',
    'processed_path': r'data\processed',
    'eda_output_path': r'outputs\eda_plots',
    'model_output_path': r'outputs\models',
    
    # === IMAGE SETTINGS ===
    'target_size': (128, 128),        # Resize all images to this
    'color_space': 'RGB',             # Keep RGB
    'image_extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'},
    
    # === PROCESSING SETTINGS ===
    'test_size': 0.20,                # 80:20 train-test split
    'random_state': 42,               # Reproducibility
    'augmentation_target': 3000,      # Target images per class after augmentation
    'max_eda_samples': 500,           # Max images per class for detailed EDA (speed)
    
    # === FEATURE EXTRACTION ===
    'hog_orientations': 9,
    'hog_pixels_per_cell': (16, 16),
    'hog_cells_per_block': (2, 2),
    'lbp_radius': 3,
    'lbp_n_points': 24,              # 8 * radius
    'color_hist_bins': 32,
    'glcm_distances': [1, 3],
    'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
}

# === CLASS MAPPING (Merging similar classes) ===
# Dataset 1 "Eksim" = Eczema in Dataset 2
# Dataset 1 "Panu" = Tinea Versicolor (kept separate as unique class)
CLASS_MAPPING = {
    # Dataset 1 classes
    'Acne': 'Acne',
    'Eksim': 'Eczema',              # Merge with Dataset 2's Eczema
    'Herpes': 'Herpes',
    'Panu': 'Panu',
    'Rosacea': 'Rosacea',
    # Dataset 2 classes
    'Basal Cell Carcinoma': 'Basal Cell Carcinoma',
    'Eczema': 'Eczema',
    'Melanocytic Nevi': 'Melanocytic Nevi',
    'Melanoma Skin Cancer': 'Melanoma Skin Cancer',
    'Seborrheic Keratoses': 'Seborrheic Keratoses',
    'Tinea Ringworm Candidiasis': 'Tinea Ringworm Candidiasis',
    'Warts Molluscum': 'Warts Molluscum',
    'Benign Keratosis': 'Benign Keratosis',
    'Psoriasis': 'Psoriasis',
    'Atopic Dermatitis': 'Atopic Dermatitis',
}

print("=" * 70)
print("SKIN DISEASE CLASSIFICATION - ML PIPELINE")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================================================
# STEP 1: UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create all necessary output directories."""
    dirs = [
        CONFIG['combined_path'],
        CONFIG['augmented_path'],
        CONFIG['processed_path'],
        CONFIG['eda_output_path'],
        CONFIG['model_output_path'],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[✓] All directories created/verified.")


def get_image_paths_and_labels(dataset_path, dataset_name=""):
    """
    Scan a dataset directory and return list of (image_path, original_label).
    Expects folder structure: dataset_path/class_name/image.jpg
    """
    data = []
    if not os.path.exists(dataset_path):
        print(f"[✗] WARNING: Path does not exist: {dataset_path}")
        return data
    
    for class_folder in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        for img_file in os.listdir(class_path):
            ext = os.path.splitext(img_file)[1].lower()
            if ext in CONFIG['image_extensions']:
                img_path = os.path.join(class_path, img_file)
                # Apply class mapping
                mapped_label = CLASS_MAPPING.get(class_folder, class_folder)
                data.append({
                    'image_path': img_path,
                    'original_label': class_folder,
                    'mapped_label': mapped_label,
                    'dataset_source': dataset_name,
                    'filename': img_file
                })
    
    return data


def load_image_safe(path, target_size=None):
    """
    Safely load an image. Returns None if corrupt/unreadable.
    
    Why we use this: Some images in datasets can be corrupt, truncated,
    or in unexpected formats. This prevents crashes during batch processing.
    """
    try:
        # Method 1: Try with PIL (better format support)
        img_pil = Image.open(path)
        img_pil.verify()  # Verify it's not corrupt
        
        # Reopen after verify (verify closes the file)
        img_pil = Image.open(path)
        img_pil = img_pil.convert('RGB')  # Ensure 3 channels
        
        # Convert to numpy array (OpenCV format for processing)
        img = np.array(img_pil)
        
        # Resize if target size specified
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        return img
    
    except Exception as e:
        return None


def compute_image_hash(img_path):
    """
    Compute MD5 hash of image file for duplicate detection.
    
    Concept: MD5 creates a unique 'fingerprint' of file contents.
    Identical files produce identical hashes.
    """
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


# ============================================================================
# STEP 2: DATASET LOADING AND INVENTORY
# ============================================================================

def load_dataset_inventory():
    """
    WHAT: Scan both datasets and create a complete inventory.
    WHY: Before any processing, we need to know exactly what we have.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATASET LOADING & INVENTORY")
    print("=" * 70)
    
    # Load Dataset 1
    print(f"\n[*] Scanning Dataset 1: {CONFIG['dataset1_path']}")
    data1 = get_image_paths_and_labels(CONFIG['dataset1_path'], "Dataset1")
    print(f"    Found {len(data1)} images")
    
    # Load Dataset 2
    print(f"[*] Scanning Dataset 2: {CONFIG['dataset2_path']}")
    data2 = get_image_paths_and_labels(CONFIG['dataset2_path'], "Dataset2")
    print(f"    Found {len(data2)} images")
    
    # Combine into DataFrame
    all_data = data1 + data2
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("\n[✗] ERROR: No images found! Check your dataset paths in CONFIG.")
        print(f"    Dataset 1 path: {os.path.abspath(CONFIG['dataset1_path'])}")
        print(f"    Dataset 2 path: {os.path.abspath(CONFIG['dataset2_path'])}")
        sys.exit(1)
    
    print(f"\n[✓] TOTAL COMBINED: {len(df)} images")
    print(f"[✓] UNIQUE CLASSES (after mapping): {df['mapped_label'].nunique()}")
    print(f"\n{'Class':<35} {'Count':>6} {'Source'}")
    print("-" * 55)
    
    for label in sorted(df['mapped_label'].unique()):
        subset = df[df['mapped_label'] == label]
        count = len(subset)
        sources = ', '.join(subset['dataset_source'].unique())
        print(f"  {label:<33} {count:>6}   ({sources})")
    
    return df


# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def eda_class_distribution(df):
    """
    EDA Step 1: Class Distribution Analysis
    
    CONCEPT: Class imbalance means some diseases have many more images than
    others. If Melanocytic Nevi has 7960 images but Panu has only 297,
    the model will learn to predict "Melanocytic Nevi" by default because
    it's right most of the time — this is the "accuracy paradox."
    
    We visualize this to decide on augmentation strategy later.
    """
    print("\n" + "-" * 50)
    print("EDA 1: CLASS DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    class_counts = df['mapped_label'].value_counts().sort_values(ascending=True)
    
    # Statistics
    print(f"\n  Total images: {len(df)}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Largest class: {class_counts.idxmax()} ({class_counts.max()} images)")
    print(f"  Smallest class: {class_counts.idxmin()} ({class_counts.min()} images)")
    print(f"  Imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1")
    print(f"  Mean images/class: {class_counts.mean():.0f}")
    print(f"  Std images/class: {class_counts.std():.0f}")
    
    # Plot 1: Horizontal bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Bar chart
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(class_counts)))
    bars = axes[0].barh(range(len(class_counts)), class_counts.values, color=colors)
    axes[0].set_yticks(range(len(class_counts)))
    axes[0].set_yticklabels(class_counts.index, fontsize=10)
    axes[0].set_xlabel('Number of Images', fontsize=12)
    axes[0].set_title('Class Distribution (Combined Dataset)', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for i, (count, bar) in enumerate(zip(class_counts.values, bars)):
        axes[0].text(count + 50, i, str(count), va='center', fontsize=9)
    
    # Add mean line
    axes[0].axvline(x=class_counts.mean(), color='red', linestyle='--', 
                     label=f'Mean: {class_counts.mean():.0f}')
    axes[0].legend(fontsize=10)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, 
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '01_class_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\n  [✓] Plot saved to {CONFIG['eda_output_path']}/01_class_distribution.png")
    
    return class_counts


def eda_image_dimensions(df):
    """
    EDA Step 2: Image Dimension Analysis
    
    CONCEPT: ML models need fixed-size inputs. If your images are 
    100×100, 500×500, and 1000×1000, you MUST resize them all to 
    the same size (e.g., 128×128). 
    
    But choosing the right size matters:
    - Too small (32×32): Lose fine details like skin texture
    - Too large (512×512): Slow processing, more memory, may overfit
    - Sweet spot for traditional ML: 128×128 or 224×224
    
    We also check aspect ratios — heavily distorted images after 
    resizing may lose important features.
    """
    print("\n" + "-" * 50)
    print("EDA 2: IMAGE DIMENSION ANALYSIS")
    print("-" * 50)
    
    widths, heights, channels, aspects = [], [], [], []
    corrupt_count = 0
    corrupt_files = []
    
    # Sample for speed (checking ALL 28K images takes time)
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    
    print(f"\n  Analyzing dimensions of {len(sample_df)} sampled images...")
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="  Reading"):
        try:
            img = Image.open(row['image_path'])
            w, h = img.size
            c = len(img.getbands())
            widths.append(w)
            heights.append(h)
            channels.append(c)
            aspects.append(w / h if h > 0 else 0)
        except:
            corrupt_count += 1
            corrupt_files.append(row['image_path'])
    
    # Statistics
    print(f"\n  Width  — Min: {min(widths)}, Max: {max(widths)}, "
          f"Mean: {np.mean(widths):.0f}, Median: {np.median(widths):.0f}")
    print(f"  Height — Min: {min(heights)}, Max: {max(heights)}, "
          f"Mean: {np.mean(heights):.0f}, Median: {np.median(heights):.0f}")
    print(f"  Channels — Unique: {set(channels)}")
    print(f"  Aspect Ratios — Mean: {np.mean(aspects):.2f}, "
          f"Std: {np.std(aspects):.2f}")
    print(f"  Corrupt/Unreadable: {corrupt_count}")
    
    if corrupt_files:
        print(f"  Corrupt files (first 5):")
        for f in corrupt_files[:5]:
            print(f"    - {f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(widths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(widths):.0f}')
    axes[0, 0].set_title('Width Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].legend()
    
    axes[0, 1].hist(heights, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--',
                        label=f'Mean: {np.mean(heights):.0f}')
    axes[0, 1].set_title('Height Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].legend()
    
    axes[1, 0].scatter(widths, heights, alpha=0.3, s=10, color='purple')
    axes[1, 0].set_title('Width vs Height', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Width')
    axes[1, 0].set_ylabel('Height')
    axes[1, 0].plot([0, max(widths)], [0, max(heights)], 'r--', alpha=0.5, label='1:1 ratio')
    axes[1, 0].legend()
    
    axes[1, 1].hist(aspects, bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
    axes[1, 1].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 1].legend()
    
    plt.suptitle('Image Dimension Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '02_image_dimensions.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  [✓] Plot saved.")
    
    return corrupt_files


def eda_pixel_analysis(df):
    """
    EDA Step 3: Pixel Intensity Analysis
    
    CONCEPT: Each pixel has R, G, B values (0-255). The distribution of 
    these values tells us about image brightness, contrast, and color bias.
    
    - Mean pixel value ≈ 128: Well-exposed images
    - Mean pixel value < 80: Dark images (may need brightness correction)
    - Mean pixel value > 200: Overexposed images
    
    Different classes may have different color profiles:
    - Rosacea: Higher red channel values
    - Melanoma: Darker overall (lower pixel values)
    
    This helps decide if we need color normalization.
    """
    print("\n" + "-" * 50)
    print("EDA 3: PIXEL INTENSITY ANALYSIS")
    print("-" * 50)
    
    class_pixel_stats = {}
    
    for label in sorted(df['mapped_label'].unique()):
        class_df = df[df['mapped_label'] == label]
        sample = class_df.sample(min(100, len(class_df)), random_state=42)
        
        r_means, g_means, b_means = [], [], []
        overall_means, overall_stds = [], []
        
        for _, row in sample.iterrows():
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            if img is not None:
                r_means.append(img[:, :, 0].mean())
                g_means.append(img[:, :, 1].mean())
                b_means.append(img[:, :, 2].mean())
                overall_means.append(img.mean())
                overall_stds.append(img.std())
        
        if r_means:
            class_pixel_stats[label] = {
                'R_mean': np.mean(r_means),
                'G_mean': np.mean(g_means),
                'B_mean': np.mean(b_means),
                'Overall_mean': np.mean(overall_means),
                'Overall_std': np.mean(overall_stds),
            }
    
    # Display stats
    stats_df = pd.DataFrame(class_pixel_stats).T
    print(f"\n  Per-Class Pixel Statistics:")
    print(stats_df.round(2).to_string())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RGB means per class
    x = range(len(stats_df))
    width = 0.25
    axes[0].bar([i - width for i in x], stats_df['R_mean'], width, 
                color='red', alpha=0.7, label='Red')
    axes[0].bar(x, stats_df['G_mean'], width, 
                color='green', alpha=0.7, label='Green')
    axes[0].bar([i + width for i in x], stats_df['B_mean'], width, 
                color='blue', alpha=0.7, label='Blue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats_df.index, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean Pixel Value')
    axes[0].set_title('RGB Channel Means per Class', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 255)
    
    # Overall mean and std
    axes[1].errorbar(range(len(stats_df)), stats_df['Overall_mean'], 
                     yerr=stats_df['Overall_std'], fmt='o', capsize=5,
                     color='purple', markersize=8)
    axes[1].set_xticks(range(len(stats_df)))
    axes[1].set_xticklabels(stats_df.index, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Pixel Value')
    axes[1].set_title('Mean ± Std Pixel Intensity per Class', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 255)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '03_pixel_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  [✓] Plot saved.")
    
    return stats_df


def eda_sample_visualization(df):
    """
    EDA Step 4: Sample Image Visualization
    
    CONCEPT: Visual inspection is crucial for image datasets because:
    1. You can spot labeling errors (wrong class)
    2. See data quality issues (watermarks, borders, text overlays)
    3. Understand visual similarity between classes
    4. Assess if the problem is feasible for ML
    
    This is something NO automated metric can replace — human eyes 
    catch things algorithms miss.
    """
    print("\n" + "-" * 50)
    print("EDA 4: SAMPLE IMAGE VISUALIZATION")
    print("-" * 50)
    
    classes = sorted(df['mapped_label'].unique())
    n_classes = len(classes)
    samples_per_class = 4
    
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                              figsize=(3 * samples_per_class, 3 * n_classes))
    
    for i, cls in enumerate(classes):
        class_df = df[df['mapped_label'] == cls]
        samples = class_df.sample(min(samples_per_class, len(class_df)), random_state=42)
        
        for j, (_, row) in enumerate(samples.iterrows()):
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            
            if n_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i][j] if samples_per_class > 1 else axes[i]
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'CORRUPT', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{cls}\n({len(class_df)} imgs)', fontsize=9, 
                           fontweight='bold')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '04_sample_images.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  [✓] Sample visualization saved.")


def eda_color_histograms(df):
    """
    EDA Step 5: Color Distribution per Class
    
    CONCEPT: A color histogram shows the frequency of each pixel 
    intensity value. Comparing histograms across classes reveals:
    
    - If classes are color-separable (good for classification)
    - Which channels carry the most discriminative information
    - Whether preprocessing like histogram equalization is needed
    """
    print("\n" + "-" * 50)
    print("EDA 5: COLOR HISTOGRAM ANALYSIS")
    print("-" * 50)
    
    classes = sorted(df['mapped_label'].unique())
    n_classes = len(classes)
    
    fig, axes = plt.subplots(
        (n_classes + 2) // 3, 3,
        figsize=(15, 4 * ((n_classes + 2) // 3))
    )
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        class_df = df[df['mapped_label'] == cls]
        sample = class_df.sample(min(50, len(class_df)), random_state=42)
        
        r_hist = np.zeros(256)
        g_hist = np.zeros(256)
        b_hist = np.zeros(256)
        count = 0
        
        for _, row in sample.iterrows():
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            if img is not None:
                r_hist += np.histogram(img[:, :, 0], bins=256, range=(0, 256))[0]
                g_hist += np.histogram(img[:, :, 1], bins=256, range=(0, 256))[0]
                b_hist += np.histogram(img[:, :, 2], bins=256, range=(0, 256))[0]
                count += 1
        
        if count > 0:
            # Normalize
            r_hist /= count
            g_hist /= count
            b_hist /= count
            
            axes[i].plot(r_hist, color='red', alpha=0.7, linewidth=0.8)
            axes[i].plot(g_hist, color='green', alpha=0.7, linewidth=0.8)
            axes[i].plot(b_hist, color='blue', alpha=0.7, linewidth=0.8)
            axes[i].set_title(cls, fontsize=9, fontweight='bold')
            axes[i].set_xlim(0, 255)
            axes[i].tick_params(labelsize=7)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Average Color Histograms per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '05_color_histograms.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  [✓] Color histograms saved.")


def eda_duplicate_detection(df):
    """
    EDA Step 6: Duplicate Image Detection
    
    CONCEPT: Duplicates in your dataset are dangerous because:
    
    1. If the same image appears in BOTH train and test sets, the model 
       essentially "memorizes" the answer — this is called DATA LEAKAGE
    2. Duplicates inflate class counts artificially
    3. They waste computation during training
    
    We use MD5 hashing: each file gets a unique fingerprint. Identical 
    files produce identical fingerprints.
    """
    print("\n" + "-" * 50)
    print("EDA 6: DUPLICATE DETECTION")
    print("-" * 50)
    
    print(f"\n  Computing hashes for {len(df)} images...")
    
    hashes = {}
    duplicates = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Hashing"):
        h = compute_image_hash(row['image_path'])
        if h:
            if h in hashes:
                duplicates.append({
                    'original': hashes[h],
                    'duplicate': row['image_path'],
                    'label': row['mapped_label']
                })
            else:
                hashes[h] = row['image_path']
    
    print(f"\n  Total unique images: {len(hashes)}")
    print(f"  Duplicate images found: {len(duplicates)}")
    
    if duplicates:
        dup_df = pd.DataFrame(duplicates)
        print(f"  Duplicates per class:")
        print(dup_df['label'].value_counts().to_string())
        
        # Save duplicate list for reference
        dup_df.to_csv(os.path.join(CONFIG['eda_output_path'], 'duplicates.csv'), index=False)
        print(f"  [✓] Duplicate list saved to duplicates.csv")
    
    return duplicates


def eda_dataset_source_analysis(df):
    """
    EDA Step 7: Dataset Source Comparison
    
    CONCEPT: Since we're merging TWO different datasets, we need to check 
    if they have systematic differences (different cameras, lighting, 
    image quality). This is called DOMAIN SHIFT — when training data 
    looks different from test data, model performance drops.
    """
    print("\n" + "-" * 50)
    print("EDA 7: DATASET SOURCE COMPARISON")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Images per source
    source_counts = df['dataset_source'].value_counts()
    axes[0].bar(source_counts.index, source_counts.values, 
                color=['steelblue', 'coral'])
    axes[0].set_title('Images per Dataset Source', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images')
    for i, (idx, val) in enumerate(source_counts.items()):
        axes[0].text(i, val + 100, str(val), ha='center', fontweight='bold')
    
    # Classes per source
    source_class = df.groupby(['dataset_source', 'mapped_label']).size().unstack(fill_value=0)
    source_class.T.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Class Distribution by Source', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Disease Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].legend(title='Source')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '06_source_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  [✓] Source comparison saved.")


def run_complete_eda(df):
    """Run all EDA steps."""
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # 1. Class distribution
    class_counts = eda_class_distribution(df)
    
    # 2. Image dimensions
    corrupt_files = eda_image_dimensions(df)
    
    # 3. Pixel analysis
    pixel_stats = eda_pixel_analysis(df)
    
    # 4. Sample visualization
    eda_sample_visualization(df)
    
    # 5. Color histograms
    eda_color_histograms(df)
    
    # 6. Duplicate detection
    duplicates = eda_duplicate_detection(df)
    
    # 7. Source comparison
    eda_dataset_source_analysis(df)
    
    # Save EDA summary
    eda_summary = {
        'total_images': len(df),
        'num_classes': df['mapped_label'].nunique(),
        'classes': sorted(df['mapped_label'].unique().tolist()),
        'class_counts': class_counts.to_dict(),
        'corrupt_files_count': len(corrupt_files),
        'duplicate_count': len(duplicates),
        'imbalance_ratio': float(class_counts.max() / class_counts.min()),
    }
    
    with open(os.path.join(CONFIG['eda_output_path'], 'eda_summary.json'), 'w') as f:
        json.dump(eda_summary, f, indent=2)
    
    print(f"\n[✓] EDA COMPLETE — All plots saved to '{CONFIG['eda_output_path']}'")
    
    return corrupt_files, duplicates


# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================

def remove_corrupt_and_duplicates(df, corrupt_files, duplicates):
    """
    PREPROCESSING Step 1: Clean the dataset
    
    CONCEPT: "Garbage in, garbage out" — corrupt images will cause 
    errors and duplicates will cause data leakage. Remove both BEFORE 
    any further processing.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)
    
    print("\n--- 3a: Removing Corrupt & Duplicate Images ---")
    
    initial_count = len(df)
    
    # Remove corrupt files
    if corrupt_files:
        df = df[~df['image_path'].isin(corrupt_files)]
        print(f"  Removed {initial_count - len(df)} corrupt images")
    
    # Remove duplicate files (keep first occurrence)
    if duplicates:
        dup_paths = [d['duplicate'] for d in duplicates]
        before = len(df)
        df = df[~df['image_path'].isin(dup_paths)]
        print(f"  Removed {before - len(df)} duplicate images")
    
    print(f"  Remaining: {len(df)} images")
    
    return df.reset_index(drop=True)


def preprocess_images(df):
    """
    PREPROCESSING Step 2: Resize, Normalize, and Save
    
    CONCEPTS:
    
    1. RESIZING: All images must be the same size for ML.
       - We use 128×128 — good balance of detail and speed for traditional ML
       - cv2.INTER_AREA: Best interpolation for shrinking images (anti-aliasing)
       - cv2.INTER_LINEAR: Good for enlarging images
    
    2. NORMALIZATION: Scale pixel values from 0-255 to 0.0-1.0
       - WHY? ML algorithms work better when features are on similar scales
       - Without normalization, a pixel value of 255 would dominate over 1
       - Division by 255.0 is called Min-Max normalization
    
    3. QUALITY CHECK: Verify each image loads correctly and has 3 channels
    """
    print("\n--- 3b: Image Preprocessing (Resize + Normalize) ---")
    print(f"  Target size: {CONFIG['target_size']}")
    print(f"  Normalization: 0-255 → 0.0-1.0")
    
    processed_images = []
    processed_labels = []
    processed_paths = []
    failed = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        img = load_image_safe(row['image_path'], CONFIG['target_size'])
        
        if img is not None and img.shape == (CONFIG['target_size'][1], 
                                               CONFIG['target_size'][0], 3):
            # Normalize to 0-1 range
            img_normalized = img.astype(np.float32) / 255.0
            
            processed_images.append(img_normalized)
            processed_labels.append(row['mapped_label'])
            processed_paths.append(row['image_path'])
        else:
            failed += 1
    
    print(f"\n  Successfully processed: {len(processed_images)}")
    print(f"  Failed: {failed}")
    
    images_array = np.array(processed_images)
    labels_array = np.array(processed_labels)
    
    print(f"  Images array shape: {images_array.shape}")
    print(f"  Memory usage: {images_array.nbytes / (1024**3):.2f} GB")
    
    return images_array, labels_array, processed_paths


# ============================================================================
# STEP 5: DATA AUGMENTATION
# ============================================================================

def augment_image(image):
    """
    Apply random augmentations to a single image.
    
    CONCEPT: Data Augmentation creates new training images by applying 
    random transformations to existing ones. This:
    
    1. INCREASES DATASET SIZE: More data = better generalization
    2. REDUCES OVERFITTING: Model sees varied versions of each image
    3. HANDLES CLASS IMBALANCE: Generate more images for minority classes
    4. SIMULATES REAL-WORLD VARIATION: Different angles, lighting, etc.
    
    AUGMENTATION TYPES EXPLAINED:
    
    - Horizontal Flip: Mirror image left-right. Skin lesions look the 
      same regardless of orientation.
    
    - Rotation: Rotate by small angle (±20°). A mole is a mole whether 
      viewed straight or at an angle.
    
    - Brightness: Randomly adjust brightness. Real photos have varying 
      lighting conditions.
    
    - Zoom/Scale: Slightly zoom in/out. Different camera distances.
    
    - Gaussian Noise: Add random noise. Simulates camera sensor noise 
      and image compression artifacts.
    
    NOTE: We do NOT use vertical flips for skin images as clinical photos 
    are typically taken with consistent vertical orientation.
    """
    augmented = image.copy()
    
    # Ensure image is in correct format
    if augmented.max() <= 1.0:
        augmented = (augmented * 255).astype(np.uint8)
    
    # Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # Random rotation (-20 to +20 degrees)
    if np.random.random() > 0.3:
        angle = np.random.uniform(-20, 20)
        h, w = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness adjustment
    if np.random.random() > 0.3:
        factor = np.random.uniform(0.7, 1.3)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(augmented)
        augmented = np.clip((augmented - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    # Random Gaussian noise
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 10, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Random zoom (crop and resize)
    if np.random.random() > 0.5:
        h, w = augmented.shape[:2]
        crop_factor = np.random.uniform(0.85, 1.0)
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_y = np.random.randint(0, h - new_h + 1)
        start_x = np.random.randint(0, w - new_w + 1)
        augmented = augmented[start_y:start_y + new_h, start_x:start_x + new_w]
        augmented = cv2.resize(augmented, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize back to 0-1
    augmented = augmented.astype(np.float32) / 255.0
    
    return augmented


def balance_dataset_with_augmentation(images, labels):
    """
    AUGMENTATION STRATEGY: Balance all classes to have equal representation.
    
    CONCEPT: Instead of simple oversampling (copying existing images), 
    we create NEW variations through augmentation. This gives the model 
    more diverse examples for minority classes.
    
    Strategy:
    1. Find the target count (we use CONFIG['augmentation_target'])
    2. For classes with fewer images: generate augmented copies until target
    3. For classes with more images: keep original (or subsample if too many)
    """
    print("\n" + "=" * 70)
    print("STEP 4: DATA AUGMENTATION")
    print("=" * 70)
    
    unique_labels = np.unique(labels)
    target_count = CONFIG['augmentation_target']
    
    print(f"\n  Target images per class: {target_count}")
    print(f"\n  {'Class':<35} {'Current':>7} {'Action':>10} {'After':>7}")
    print("  " + "-" * 65)
    
    augmented_images = []
    augmented_labels = []
    
    for label in unique_labels:
        mask = labels == label
        class_images = images[mask]
        current_count = len(class_images)
        
        # Add all original images
        augmented_images.extend(class_images)
        augmented_labels.extend([label] * current_count)
        
        if current_count < target_count:
            # Need to augment
            needed = target_count - current_count
            action = f"+{needed}"
            
            for i in range(needed):
                # Pick a random original image to augment
                idx = np.random.randint(0, current_count)
                aug_img = augment_image(class_images[idx])
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        else:
            action = "keep"
            needed = 0
        
        final_count = current_count + max(0, target_count - current_count)
        print(f"  {label:<35} {current_count:>7} {action:>10} {final_count:>7}")
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    print(f"\n  Total images after augmentation: {len(augmented_images)}")
    print(f"  Array shape: {augmented_images.shape}")
    
    # Visualize augmentation examples
    visualize_augmentation(images, labels)
    
    return augmented_images, augmented_labels


def visualize_augmentation(images, labels):
    """Show original vs augmented examples."""
    print("\n  Generating augmentation visualization...")
    
    unique_labels = np.unique(labels)
    n_show = min(4, len(unique_labels))
    
    fig, axes = plt.subplots(n_show, 5, figsize=(15, 3 * n_show))
    
    for i in range(n_show):
        label = unique_labels[i]
        mask = labels == label
        class_images = images[mask]
        
        # Original
        original = class_images[0]
        axes[i][0].imshow(original)
        axes[i][0].set_title(f'{label}\n(Original)', fontsize=9)
        axes[i][0].axis('off')
        
        # 4 augmented versions
        for j in range(1, 5):
            aug = augment_image(original)
            axes[i][j].imshow(aug)
            axes[i][j].set_title(f'Augmented {j}', fontsize=9)
            axes[i][j].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['eda_output_path'], '07_augmentation_examples.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================================
# STEP 6: FEATURE EXTRACTION
# ============================================================================

def extract_hog_features(image):
    """
    HOG (Histogram of Oriented Gradients) Features
    
    CONCEPT: HOG captures the SHAPE and EDGE information in an image.
    
    How it works:
    1. Divide image into small cells (e.g., 16×16 pixels)
    2. In each cell, compute gradient magnitude and direction
    3. Create a histogram of gradient orientations (9 bins = 9 directions)
    4. Normalize across blocks of cells for lighting invariance
    
    WHY FOR SKIN DISEASE: Skin lesions have characteristic shapes and 
    border patterns. Melanoma has irregular borders, while benign moles 
    have smooth borders. HOG captures these differences.
    
    Output: 1D vector of numbers representing edge/shape patterns
    """
    gray = rgb2gray(image)  # Convert to grayscale for edge detection
    
    features = hog(
        gray,
        orientations=CONFIG['hog_orientations'],
        pixels_per_cell=CONFIG['hog_pixels_per_cell'],
        cells_per_block=CONFIG['hog_cells_per_block'],
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features


def extract_lbp_features(image):
    """
    LBP (Local Binary Pattern) Features
    
    CONCEPT: LBP captures TEXTURE information.
    
    How it works:
    1. For each pixel, look at its neighbors in a circle (radius=3)
    2. Compare: if neighbor > center pixel, write 1; else write 0
    3. The binary pattern (e.g., 01101010) encodes local texture
    4. Create histogram of all patterns = texture descriptor
    
    WHY FOR SKIN DISEASE: Different skin conditions have distinct 
    textures. Psoriasis has scaly texture, eczema has rough patches, 
    melanocytic nevi have specific patterns. LBP quantifies these.
    
    Output: Histogram of texture patterns
    """
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    
    lbp = local_binary_pattern(
        gray,
        P=CONFIG['lbp_n_points'],
        R=CONFIG['lbp_radius'],
        method='uniform'
    )
    
    # Create histogram of LBP values
    n_bins = CONFIG['lbp_n_points'] + 2  # uniform LBP has P+2 unique values
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_color_histogram(image):
    """
    Color Histogram Features
    
    CONCEPT: Color histograms describe the COLOR DISTRIBUTION of an image.
    
    How it works:
    1. For each channel (R, G, B), count pixels at each intensity level
    2. Use 32 bins (instead of 256) to reduce dimensionality
    3. Normalize so total = 1 (makes it independent of image size)
    
    WHY FOR SKIN DISEASE: Skin conditions have characteristic colors.
    - Rosacea: More red tones
    - Melanoma: Darker, multiple colors
    - Vitiligo: Lighter patches
    - Normal skin: Balanced RGB
    
    Output: Concatenated R, G, B histograms (32 × 3 = 96 values)
    """
    n_bins = CONFIG['color_hist_bins']
    histograms = []
    
    for channel in range(3):  # R, G, B
        hist, _ = np.histogram(
            image[:, :, channel].ravel(),
            bins=n_bins,
            range=(0, 1),  # Images are normalized to 0-1
            density=True
        )
        histograms.extend(hist)
    
    return np.array(histograms)


def extract_glcm_features(image):
    """
    GLCM (Gray-Level Co-occurrence Matrix) Features
    
    CONCEPT: GLCM analyzes how often pairs of pixel intensities occur 
    next to each other. It captures TEXTURE REGULARITY.
    
    How it works:
    1. For each pixel pair at distance d and angle θ, count co-occurrences
    2. Build a matrix where entry (i,j) = how often intensity i appears 
       next to intensity j
    3. Compute statistical properties of this matrix:
       - Contrast: Intensity difference between neighbors (rough vs smooth)
       - Correlation: Linear dependency of intensities
       - Energy: Uniformity/homogeneity of texture
       - Homogeneity: Closeness of distribution to GLCM diagonal
       - Dissimilarity: Variation of gray levels
       - ASM (Angular Second Moment): Orderliness of texture
    
    WHY FOR SKIN DISEASE: Malignant lesions tend to have irregular, 
    heterogeneous texture (high contrast, low homogeneity), while benign 
    conditions are more uniform.
    
    Output: Statistical measures of texture patterns
    """
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    
    # Reduce gray levels for computational efficiency (256 → 32 levels)
    gray = (gray // 8).astype(np.uint8)
    
    glcm = graycomatrix(
        gray,
        distances=CONFIG['glcm_distances'],
        angles=CONFIG['glcm_angles'],
        levels=32,
        symmetric=True,
        normed=True
    )
    
    # Extract properties
    properties = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
    features = []
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend([values.mean(), values.std()])
    
    return np.array(features)


def extract_statistical_features(image):
    """
    Basic Statistical Features
    
    CONCEPT: Simple statistics of pixel values capture global image 
    properties without complex algorithms.
    
    Features per channel:
    - Mean: Average brightness
    - Std: Contrast
    - Skewness: Asymmetry of pixel distribution
    - Kurtosis: Peakedness of pixel distribution
    - Min, Max: Dynamic range
    
    WHY: These are cheap to compute and provide baseline discriminative 
    power. Combined with other features, they improve overall performance.
    """
    features = []
    
    for channel in range(3):
        ch = image[:, :, channel].ravel()
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.median(ch),
            float(pd.Series(ch).skew()),
            float(pd.Series(ch).kurtosis()),
            np.percentile(ch, 25),
            np.percentile(ch, 75),
        ])
    
    return np.array(features)


def extract_all_features(image):
    """
    Combine ALL feature extractors into a single feature vector.
    
    This is called FEATURE FUSION — using multiple complementary 
    feature types gives a richer representation than any single type.
    
    Total features per image:
    - HOG: ~1764 (depends on image size and cell size)
    - LBP: 26 (n_points + 2)
    - Color Histogram: 96 (32 bins × 3 channels)
    - GLCM: 10 (5 properties × 2 stats each)
    - Statistical: 21 (7 stats × 3 channels)
    """
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    color_feat = extract_color_histogram(image)
    glcm_feat = extract_glcm_features(image)
    stat_feat = extract_statistical_features(image)
    
    # Concatenate all features into one vector
    combined = np.concatenate([hog_feat, lbp_feat, color_feat, glcm_feat, stat_feat])
    
    return combined


def extract_features_from_dataset(images, labels):
    """
    Extract features from all images and create ML-ready dataset.
    
    This is the bridge between IMAGE DATA and TABULAR DATA that 
    traditional ML algorithms can process.
    
    Input: Array of images (N, 128, 128, 3)
    Output: Feature matrix (N, num_features)
    """
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE EXTRACTION")
    print("=" * 70)
    
    # First, extract from one image to get feature size
    sample_features = extract_all_features(images[0])
    n_features = len(sample_features)
    print(f"\n  Features per image: {n_features}")
    print(f"  Feature breakdown:")
    
    # Calculate individual feature sizes
    hog_size = len(extract_hog_features(images[0]))
    lbp_size = len(extract_lbp_features(images[0]))
    color_size = len(extract_color_histogram(images[0]))
    glcm_size = len(extract_glcm_features(images[0]))
    stat_size = len(extract_statistical_features(images[0]))
    
    print(f"    HOG:             {hog_size}")
    print(f"    LBP:             {lbp_size}")
    print(f"    Color Histogram: {color_size}")
    print(f"    GLCM:            {glcm_size}")
    print(f"    Statistical:     {stat_size}")
    print(f"    TOTAL:           {n_features}")
    
    # Extract features for all images
    print(f"\n  Extracting features from {len(images)} images...")
    
    feature_matrix = np.zeros((len(images), n_features), dtype=np.float32)
    failed_indices = []
    
    for i in tqdm(range(len(images)), desc="  Extracting"):
        try:
            feature_matrix[i] = extract_all_features(images[i])
        except Exception as e:
            failed_indices.append(i)
            # Fill with zeros for failed extractions
            feature_matrix[i] = np.zeros(n_features)
    
    if failed_indices:
        print(f"\n  WARNING: Feature extraction failed for {len(failed_indices)} images")
        # Remove failed images
        mask = np.ones(len(images), dtype=bool)
        mask[failed_indices] = False
        feature_matrix = feature_matrix[mask]
        labels = labels[mask]
    
    print(f"\n  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Memory: {feature_matrix.nbytes / (1024**2):.1f} MB")
    
    # Check for NaN or Inf values
    nan_count = np.isnan(feature_matrix).sum()
    inf_count = np.isinf(feature_matrix).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    # Replace NaN/Inf with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_matrix, labels


# ============================================================================
# STEP 7: FINAL ML-READY DATASET PREPARATION
# ============================================================================

def prepare_ml_ready_dataset(features, labels):
    """
    Final step: Create train-test split, encode labels, scale features.
    
    CONCEPTS:
    
    1. LABEL ENCODING: Convert text labels ("Acne", "Eczema") to numbers (0, 1).
       ML algorithms need numerical inputs. LabelEncoder maps each unique 
       class to an integer.
    
    2. TRAIN-TEST SPLIT: Divide data into training (80%) and testing (20%).
       - Training set: Model learns patterns from this data
       - Test set: Evaluates model on UNSEEN data (prevents cheating)
       - stratify=labels: Ensures each class has proportional representation 
         in both train and test sets
    
    3. FEATURE SCALING (StandardScaler):
       - Transforms features to mean=0, std=1
       - WHY? SVM and KNN use distance calculations. If HOG features range 
         from 0-0.5 but color histograms range from 0-1000, the color 
         features would dominate distance calculations unfairly
       - IMPORTANT: Fit scaler on TRAINING data only, then transform both 
         train and test. Fitting on test data would be DATA LEAKAGE.
    
    4. CLASS WEIGHTS: Computed to handle any remaining imbalance.
       - Gives higher penalty for misclassifying minority classes
       - 'balanced': weight = n_samples / (n_classes × n_samples_per_class)
    """
    print("\n" + "=" * 70)
    print("STEP 6: ML-READY DATASET PREPARATION")
    print("=" * 70)
    
    # --- Label Encoding ---
    print("\n--- 6a: Label Encoding ---")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    print(f"  Classes and their codes:")
    for i, cls in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"    {i:>2}: {cls:<35} ({count} samples)")
    
    # --- Train-Test Split ---
    print("\n--- 6b: Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_encoded,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y_encoded  # IMPORTANT: maintains class proportions
    )
    
    print(f"  Training set: {X_train.shape[0]} samples ({(1-CONFIG['test_size'])*100:.0f}%)")
    print(f"  Test set:     {X_test.shape[0]} samples ({CONFIG['test_size']*100:.0f}%)")
    print(f"  Features:     {X_train.shape[1]}")
    
    # Verify stratification
    print(f"\n  Class distribution verification:")
    train_dist = np.bincount(y_train) / len(y_train) * 100
    test_dist = np.bincount(y_test) / len(y_test) * 100
    print(f"    {'Class':<35} {'Train %':>8} {'Test %':>8}")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"    {cls:<35} {train_dist[i]:>7.1f}% {test_dist[i]:>7.1f}%")
    
    # --- Feature Scaling ---
    print("\n--- 6c: Feature Scaling (StandardScaler) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit on train ONLY
    X_test_scaled = scaler.transform(X_test)          # Transform test with train params
    
    print(f"  Before scaling — Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  After scaling  — Train mean: {X_train_scaled.mean():.4f}, "
          f"std: {X_train_scaled.std():.4f}")
    
    # --- Class Weights ---
    print("\n--- 6d: Computing Class Weights ---")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"  Class weights (for handling remaining imbalance):")
    for cls_id, weight in class_weight_dict.items():
        cls_name = label_encoder.classes_[cls_id]
        print(f"    {cls_name:<35}: {weight:.4f}")
    
    # --- Save Everything ---
    print("\n--- 6e: Saving ML-Ready Dataset ---")
    
    output = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'class_weights': class_weight_dict,
        'feature_names': {
            'hog_size': len(extract_hog_features(np.zeros((128, 128, 3)))),
            'lbp_size': len(extract_lbp_features(np.zeros((128, 128, 3)))),
            'color_hist_size': CONFIG['color_hist_bins'] * 3,
            'glcm_size': 10,
            'stat_size': 21,
        },
        'config': CONFIG,
    }
    
    # Save as pickle (all-in-one)
    save_path = os.path.join(CONFIG['processed_path'], 'ml_ready_dataset.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"  [✓] Saved complete ML-ready dataset to: {save_path}")
    
    # Also save individual components as numpy files (easier to load)
    np.save(os.path.join(CONFIG['processed_path'], 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(CONFIG['processed_path'], 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(CONFIG['processed_path'], 'y_train.npy'), y_train)
    np.save(os.path.join(CONFIG['processed_path'], 'y_test.npy'), y_test)
    joblib.dump(label_encoder, os.path.join(CONFIG['processed_path'], 'label_encoder.joblib'))
    joblib.dump(scaler, os.path.join(CONFIG['processed_path'], 'scaler.joblib'))
    
    print(f"  [✓] Saved individual .npy and .joblib files")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("ML-READY DATASET SUMMARY")
    print("=" * 70)
    print(f"  Training samples:  {X_train_scaled.shape[0]}")
    print(f"  Test samples:      {X_test_scaled.shape[0]}")
    print(f"  Feature dimension: {X_train_scaled.shape[1]}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"\n  Files saved in: {os.path.abspath(CONFIG['processed_path'])}")
    
    return output


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Execute the complete pipeline.
    
    PIPELINE FLOW:
    Raw Images → EDA → Clean → Preprocess → Augment → Extract Features → ML-Ready
    """
    
    # Step 0: Create directories
    create_directories()
    
    # Step 1: Load dataset inventory
    df = load_dataset_inventory()
    
    # Step 2: Run complete EDA
    corrupt_files, duplicates = run_complete_eda(df)
    
    # Step 3: Preprocessing - Clean data
    df_clean = remove_corrupt_and_duplicates(df, corrupt_files, duplicates)
    
    # Step 3b: Preprocessing - Resize and normalize
    images, labels, paths = preprocess_images(df_clean)
    
    # Step 4: Data Augmentation
    images_aug, labels_aug = balance_dataset_with_augmentation(images, labels)
    
    # Free memory from original images
    del images
    
    # Step 5: Feature Extraction
    features, labels_final = extract_features_from_dataset(images_aug, labels_aug)
    
    # Free memory from image arrays
    del images_aug
    
    # Step 6: Prepare ML-Ready Dataset
    ml_data = prepare_ml_ready_dataset(features, labels_final)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nYour ML-ready dataset is saved at:")
    print(f"  {os.path.abspath(CONFIG['processed_path'])}")
    print(f"\nNext steps:")
    print(f"  1. Load the dataset: pickle.load('data/processed/ml_ready_dataset.pkl')")
    print(f"  2. Train models: SVM, Random Forest, KNN, Decision Tree")
    print(f"  3. Evaluate: Accuracy, Precision, Recall, F1-Score, Confusion Matrix")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()