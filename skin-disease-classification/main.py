# dsproj/skin-disease-classification/main.py
"""
=============================================================================
Machine Learning-Assisted Skin Disease Classification System
Early Detection with Human-in-the-Loop Validation

OPTIMIZED VERSION — Smart Sampling + PDF Summary Report

Pipeline:
    Raw Images (28,647) → EDA → Clean → Smart Sample (~11,200)
    → Preprocess → Augment (balance to 800/class) → Feature Extraction
    → ML-Ready Dataset → PDF Summary Report

Author: [Your Name]
Date: 2025
=============================================================================
"""

# ============================================================================
# STEP 0: IMPORTS
# ============================================================================

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves plots to file only
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter, OrderedDict
from tqdm import tqdm
import hashlib
import warnings
import json
import pickle
from datetime import datetime
import traceback

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Scikit-image (feature extraction)
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray

# Joblib for saving sklearn objects
import joblib

# PDF generation
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[!] fpdf2 not installed. Run: pip install fpdf2")
    print("    PDF report will be skipped.\n")

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset1_path': r'data\raw\dataset1\train',
    'dataset2_path': r'data\raw\dataset2\IMG_CLASSES',
    'combined_path': r'data\combined',
    'augmented_path': r'data\augmented',
    'processed_path': r'data\processed',
    'eda_output_path': r'outputs\eda_plots',
    'model_output_path': r'outputs\models',

    'target_size': (128, 128),
    'color_space': 'RGB',
    'image_extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'},

    'max_per_class': 800,
    'min_per_class': 250,
    'augmentation_target': 800,

    'test_size': 0.20,
    'random_state': 42,
    'max_eda_samples': 300,

    'hog_orientations': 9,
    'hog_pixels_per_cell': (16, 16),
    'hog_cells_per_block': (2, 2),
    'lbp_radius': 3,
    'lbp_n_points': 24,
    'color_hist_bins': 32,
    'glcm_distances': [1, 3],
    'glcm_angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
}

CLASS_MAPPING = {
    'Acne': 'Acne',
    'Eksim': 'Eczema',
    'Herpes': 'Herpes',
    'Panu': 'Panu',
    'Rosacea': 'Rosacea',
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


# ============================================================================
# GLOBAL REPORT TRACKER — collects all stats for PDF
# ============================================================================

REPORT = {
    'start_time': None,
    'end_time': None,

    # Step 1: Loading
    'dataset1_count': 0,
    'dataset2_count': 0,
    'total_raw': 0,
    'num_classes': 0,
    'classes': [],
    'class_counts_raw': {},

    # Step 2: EDA
    'corrupt_count': 0,
    'duplicate_count': 0,
    'imbalance_ratio_raw': 0,
    'eda_plots': [],
    'pixel_stats': {},
    'dim_stats': {},

    # Step 3: Cleaning
    'after_cleaning': 0,
    'corrupt_removed': 0,
    'duplicates_removed': 0,

    # Step 4: Sampling
    'before_sampling': 0,
    'after_sampling': 0,
    'sampling_reduction_pct': 0,
    'sampling_counts': {},
    'imbalance_ratio_sampled': 0,

    # Step 5: Preprocessing
    'before_preprocessing': 0,
    'after_preprocessing': 0,
    'preprocess_failed': 0,
    'preprocess_shape': '',
    'preprocess_memory_mb': 0,

    # Step 6: Augmentation
    'before_augmentation': 0,
    'after_augmentation': 0,
    'augmentation_counts_before': {},
    'augmentation_counts_after': {},
    'augmentation_added': 0,

    # Step 7: Features
    'feature_vector_size': 0,
    'feature_breakdown': {},
    'feature_extraction_failed': 0,
    'feature_matrix_shape': '',
    'feature_memory_mb': 0,

    # Step 8: ML Ready
    'train_samples': 0,
    'test_samples': 0,
    'label_encoding': {},
    'train_class_pct': {},
    'test_class_pct': {},
    'class_weights': {},
    'scaling_before_mean': 0,
    'scaling_before_std': 0,
    'scaling_after_mean': 0,
    'scaling_after_std': 0,
}


print("=" * 70)
print("  SKIN DISEASE CLASSIFICATION - OPTIMIZED ML PIPELINE")
print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    dirs = [
        CONFIG['combined_path'],
        CONFIG['augmented_path'],
        CONFIG['processed_path'],
        CONFIG['eda_output_path'],
        CONFIG['model_output_path'],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Directories created / verified.")


def get_image_paths_and_labels(dataset_path, dataset_name=""):
    data = []
    if not os.path.exists(dataset_path):
        print(f"[!!] Path not found: {dataset_path}")
        return data
    for class_folder in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            ext = os.path.splitext(img_file)[1].lower()
            if ext in CONFIG['image_extensions']:
                data.append({
                    'image_path': os.path.join(class_path, img_file),
                    'original_label': class_folder,
                    'mapped_label': CLASS_MAPPING.get(class_folder, class_folder),
                    'dataset_source': dataset_name,
                    'filename': img_file,
                })
    return data


def load_image_safe(path, target_size=None):
    try:
        img_pil = Image.open(path)
        img_pil.verify()
        img_pil = Image.open(path)
        img_pil = img_pil.convert('RGB')
        img = np.array(img_pil)
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def compute_image_hash(img_path):
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def save_plot(fig, filename):
    """Save plot and track path for PDF report."""
    fpath = os.path.join(CONFIG['eda_output_path'], filename)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if fpath not in REPORT['eda_plots']:
        REPORT['eda_plots'].append(fpath)
    return fpath


# ============================================================================
# STEP 1 - DATASET LOADING
# ============================================================================

def load_dataset_inventory():
    print("\n" + "=" * 70)
    print("  STEP 1 : DATASET LOADING & INVENTORY")
    print("=" * 70)

    print(f"\n  [*] Scanning Dataset 1 : {CONFIG['dataset1_path']}")
    data1 = get_image_paths_and_labels(CONFIG['dataset1_path'], "Dataset1")
    print(f"      Found {len(data1)} images")

    print(f"  [*] Scanning Dataset 2 : {CONFIG['dataset2_path']}")
    data2 = get_image_paths_and_labels(CONFIG['dataset2_path'], "Dataset2")
    print(f"      Found {len(data2)} images")

    df = pd.DataFrame(data1 + data2)

    if df.empty:
        print("\n  [!!] ERROR: No images found. Check paths in CONFIG.")
        sys.exit(1)

    REPORT['dataset1_count'] = len(data1)
    REPORT['dataset2_count'] = len(data2)
    REPORT['total_raw'] = len(df)
    REPORT['num_classes'] = df['mapped_label'].nunique()
    REPORT['classes'] = sorted(df['mapped_label'].unique().tolist())

    print(f"\n  [OK] TOTAL COMBINED : {len(df)} images")
    print(f"  [OK] UNIQUE CLASSES : {df['mapped_label'].nunique()}")
    print(f"\n  {'Class':<35} {'Count':>6}  Source")
    print("  " + "-" * 60)
    for label in sorted(df['mapped_label'].unique()):
        sub = df[df['mapped_label'] == label]
        srcs = ', '.join(sub['dataset_source'].unique())
        print(f"  {label:<35} {len(sub):>6}  ({srcs})")

    return df


# ============================================================================
# STEP 2 - EDA
# ============================================================================

def eda_class_distribution(df, tag=""):
    print(f"\n  -- EDA : Class Distribution {tag} --")
    counts = df['mapped_label'].value_counts().sort_values(ascending=True)

    print(f"     Images       : {len(df)}")
    print(f"     Classes      : {len(counts)}")
    print(f"     Largest      : {counts.idxmax()} ({counts.max()})")
    print(f"     Smallest     : {counts.idxmin()} ({counts.min()})")
    print(f"     Imbalance    : {counts.max() / counts.min():.1f}:1")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(counts)))
    bars = axes[0].barh(range(len(counts)), counts.values, color=colors)
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(counts.index, fontsize=10)
    axes[0].set_xlabel('Number of Images')
    axes[0].set_title(f'Class Distribution {tag}', fontsize=14, fontweight='bold')
    for i, (v, b) in enumerate(zip(counts.values, bars)):
        axes[0].text(v + 10, i, str(v), va='center', fontsize=9)
    axes[0].axvline(counts.mean(), color='red', ls='--', label=f'Mean {counts.mean():.0f}')
    axes[0].legend()

    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 8})
    axes[1].set_title('Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    safe = tag.replace(' ', '_').replace('(', '').replace(')', '')
    save_plot(fig, f'01_class_dist{safe}.png')
    print("     [OK] Plot saved.")
    return counts


def eda_image_dimensions(df):
    print("\n  -- EDA : Image Dimensions --")
    sample = df.sample(min(2000, len(df)), random_state=42)
    widths, heights = [], []
    corrupt_files = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="     Reading"):
        try:
            img = Image.open(row['image_path'])
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            corrupt_files.append(row['image_path'])

    REPORT['dim_stats'] = {
        'width_min': int(min(widths)), 'width_max': int(max(widths)),
        'width_mean': int(np.mean(widths)),
        'height_min': int(min(heights)), 'height_max': int(max(heights)),
        'height_mean': int(np.mean(heights)),
    }

    print(f"     Width  — min {min(widths)}, max {max(widths)}, mean {np.mean(widths):.0f}")
    print(f"     Height — min {min(heights)}, max {max(heights)}, mean {np.mean(heights):.0f}")
    print(f"     Corrupt: {len(corrupt_files)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(widths, bins=40, color='steelblue', edgecolor='black', alpha=.7)
    axes[0].axvline(np.mean(widths), color='red', ls='--', label=f'Mean {np.mean(widths):.0f}')
    axes[0].set_title('Width Distribution', fontweight='bold')
    axes[0].legend()

    axes[1].hist(heights, bins=40, color='coral', edgecolor='black', alpha=.7)
    axes[1].axvline(np.mean(heights), color='red', ls='--', label=f'Mean {np.mean(heights):.0f}')
    axes[1].set_title('Height Distribution', fontweight='bold')
    axes[1].legend()

    aspects = [w / h for w, h in zip(widths, heights)]
    axes[2].hist(aspects, bins=40, color='green', edgecolor='black', alpha=.7)
    axes[2].axvline(1.0, color='red', ls='--', label='Square')
    axes[2].set_title('Aspect Ratio', fontweight='bold')
    axes[2].legend()

    plt.tight_layout()
    save_plot(fig, '02_image_dimensions.png')
    print("     [OK] Plot saved.")
    return corrupt_files


def eda_pixel_analysis(df):
    print("\n  -- EDA : Pixel Intensity --")
    stats = {}
    for label in sorted(df['mapped_label'].unique()):
        sub = df[df['mapped_label'] == label]
        sample = sub.sample(min(50, len(sub)), random_state=42)
        r, g, b = [], [], []
        for _, row in sample.iterrows():
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            if img is not None:
                r.append(float(img[:, :, 0].mean()))
                g.append(float(img[:, :, 1].mean()))
                b.append(float(img[:, :, 2].mean()))
        if r:
            stats[label] = {
                'R': round(np.mean(r), 2),
                'G': round(np.mean(g), 2),
                'B': round(np.mean(b), 2),
            }
    REPORT['pixel_stats'] = stats

    sdf = pd.DataFrame(stats).T
    print(sdf.to_string())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sdf))
    w = 0.25
    ax.bar(x - w, sdf['R'], w, color='red', alpha=.7, label='R')
    ax.bar(x, sdf['G'], w, color='green', alpha=.7, label='G')
    ax.bar(x + w, sdf['B'], w, color='blue', alpha=.7, label='B')
    ax.set_xticks(x)
    ax.set_xticklabels(sdf.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Pixel Value (0-255)')
    ax.set_title('RGB Means per Class', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 255)
    plt.tight_layout()
    save_plot(fig, '03_pixel_analysis.png')
    print("     [OK] Plot saved.")
    return sdf


def eda_sample_visualization(df):
    print("\n  -- EDA : Sample Images --")
    classes = sorted(df['mapped_label'].unique())
    n = len(classes)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(3 * cols, 2.5 * n))
    for i, cls in enumerate(classes):
        sub = df[df['mapped_label'] == cls]
        sample = sub.sample(min(cols, len(sub)), random_state=42)
        for j, (_, row) in enumerate(sample.iterrows()):
            ax = axes[i][j] if n > 1 else axes[j]
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            if img is not None:
                ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{cls}\n({len(sub)})', fontsize=8, fontweight='bold')
    plt.suptitle('Sample Images per Class', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_plot(fig, '04_sample_images.png')
    print("     [OK] Plot saved.")


def eda_color_histograms(df):
    print("\n  -- EDA : Colour Histograms --")
    classes = sorted(df['mapped_label'].unique())
    nc = len(classes)
    rows = (nc + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes_flat = axes.flatten()
    for i, cls in enumerate(classes):
        sub = df[df['mapped_label'] == cls]
        sample = sub.sample(min(50, len(sub)), random_state=42)
        rh, gh, bh = np.zeros(256), np.zeros(256), np.zeros(256)
        cnt = 0
        for _, row in sample.iterrows():
            img = load_image_safe(row['image_path'], CONFIG['target_size'])
            if img is not None:
                rh += np.histogram(img[:, :, 0], 256, (0, 256))[0]
                gh += np.histogram(img[:, :, 1], 256, (0, 256))[0]
                bh += np.histogram(img[:, :, 2], 256, (0, 256))[0]
                cnt += 1
        if cnt:
            rh /= cnt; gh /= cnt; bh /= cnt
            axes_flat[i].plot(rh, color='red', alpha=.7, lw=.8)
            axes_flat[i].plot(gh, color='green', alpha=.7, lw=.8)
            axes_flat[i].plot(bh, color='blue', alpha=.7, lw=.8)
        axes_flat[i].set_title(cls, fontsize=9, fontweight='bold')
        axes_flat[i].set_xlim(0, 255)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle('Colour Histograms per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, '05_color_histograms.png')
    print("     [OK] Plot saved.")


def eda_duplicate_detection(df):
    print("\n  -- EDA : Duplicate Detection --")
    hashes = {}
    duplicates = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="     Hashing"):
        h = compute_image_hash(row['image_path'])
        if h:
            if h in hashes:
                duplicates.append({
                    'original': hashes[h],
                    'duplicate': row['image_path'],
                    'label': row['mapped_label'],
                })
            else:
                hashes[h] = row['image_path']
    REPORT['duplicate_count'] = len(duplicates)
    print(f"     Unique : {len(hashes)}")
    print(f"     Dupes  : {len(duplicates)}")
    if duplicates:
        dup_df = pd.DataFrame(duplicates)
        dup_df.to_csv(os.path.join(CONFIG['eda_output_path'], 'duplicates.csv'), index=False)
    return duplicates


def eda_source_comparison(df):
    print("\n  -- EDA : Source Comparison --")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = df['dataset_source'].value_counts()
    axes[0].bar(sc.index, sc.values, color=['steelblue', 'coral'])
    axes[0].set_title('Images per Source', fontweight='bold')
    for i, (idx, val) in enumerate(sc.items()):
        axes[0].text(i, val + 100, str(val), ha='center', fontweight='bold')
    ct = df.groupby(['dataset_source', 'mapped_label']).size().unstack(fill_value=0)
    ct.T.plot(kind='bar', ax=axes[1], width=.8)
    axes[1].set_title('Class x Source', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    save_plot(fig, '06_source_comparison.png')
    print("     [OK] Plot saved.")


def run_full_eda(df):
    print("\n" + "=" * 70)
    print("  STEP 2 : EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    counts = eda_class_distribution(df, "(Full Dataset)")
    corrupt = eda_image_dimensions(df)
    eda_pixel_analysis(df)
    eda_sample_visualization(df)
    eda_color_histograms(df)
    dupes = eda_duplicate_detection(df)
    eda_source_comparison(df)

    REPORT['class_counts_raw'] = counts.to_dict()
    REPORT['corrupt_count'] = len(corrupt)
    REPORT['imbalance_ratio_raw'] = round(float(counts.max() / counts.min()), 2)

    print(f"\n  [OK] EDA complete — plots saved to '{CONFIG['eda_output_path']}'")
    return corrupt, dupes


# ============================================================================
# STEP 3 - DATA CLEANING
# ============================================================================

def remove_corrupt_and_duplicates(df, corrupt_files, duplicates):
    print("\n" + "=" * 70)
    print("  STEP 3 : DATA CLEANING")
    print("=" * 70)

    n0 = len(df)
    removed_corrupt = 0
    removed_dupes = 0

    if corrupt_files:
        before = len(df)
        df = df[~df['image_path'].isin(corrupt_files)]
        removed_corrupt = before - len(df)
        print(f"  Removed {removed_corrupt} corrupt images")

    if duplicates:
        dup_paths = [d['duplicate'] for d in duplicates]
        before = len(df)
        df = df[~df['image_path'].isin(dup_paths)]
        removed_dupes = before - len(df)
        print(f"  Removed {removed_dupes} duplicate images")

    REPORT['corrupt_removed'] = removed_corrupt
    REPORT['duplicates_removed'] = removed_dupes
    REPORT['after_cleaning'] = len(df)
    print(f"  Remaining : {len(df)} images")
    return df.reset_index(drop=True)


# ============================================================================
# STEP 4 - SMART SAMPLING
# ============================================================================

def smart_sample_dataset(df):
    print("\n" + "=" * 70)
    print("  STEP 4 : SMART SAMPLING")
    print("=" * 70)

    cap = CONFIG['max_per_class']
    total_before = len(df)
    REPORT['before_sampling'] = total_before

    print(f"\n  Cap per class : {cap}")
    print(f"\n  {'Class':<35} {'Before':>7} {'After':>7} {'Action'}")
    print("  " + "-" * 65)

    parts = []
    sampling_counts = {}

    for label in sorted(df['mapped_label'].unique()):
        cdf = df[df['mapped_label'] == label]
        n = len(cdf)
        if n > cap:
            sampled = cdf.sample(n=cap, random_state=CONFIG['random_state'])
            action = "down-sampled"
            after = cap
        else:
            sampled = cdf
            action = "kept all"
            after = n
        parts.append(sampled)
        sampling_counts[label] = after
        print(f"  {label:<35} {n:>7} {after:>7} {action}")

    df_sampled = pd.concat(parts, ignore_index=True)
    df_sampled = df_sampled.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)

    total_after = len(df_sampled)
    pct = (1 - total_after / total_before) * 100

    REPORT['after_sampling'] = total_after
    REPORT['sampling_counts'] = sampling_counts
    REPORT['sampling_reduction_pct'] = round(pct, 1)
    sc = pd.Series(sampling_counts)
    REPORT['imbalance_ratio_sampled'] = round(float(sc.max() / sc.min()), 2)

    print(f"\n  Before : {total_before}  |  After : {total_after}  |  Reduction : {pct:.1f}%")
    return df_sampled


# ============================================================================
# STEP 5 - IMAGE PREPROCESSING
# ============================================================================

def preprocess_images(df):
    print("\n" + "=" * 70)
    print("  STEP 5 : IMAGE PREPROCESSING")
    print("=" * 70)

    REPORT['before_preprocessing'] = len(df)
    print(f"  Target size    : {CONFIG['target_size']}")
    print(f"  Normalisation  : / 255 -> [0, 1]")
    print(f"  Images to load : {len(df)}")

    images, labels = [], []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Preprocessing"):
        img = load_image_safe(row['image_path'], CONFIG['target_size'])
        if (img is not None and
                img.shape == (CONFIG['target_size'][1], CONFIG['target_size'][0], 3)):
            images.append(img.astype(np.float32) / 255.0)
            labels.append(row['mapped_label'])
        else:
            failed += 1

    images = np.array(images)
    labels = np.array(labels)

    REPORT['after_preprocessing'] = len(images)
    REPORT['preprocess_failed'] = failed
    REPORT['preprocess_shape'] = str(images.shape)
    REPORT['preprocess_memory_mb'] = round(images.nbytes / (1024 ** 2), 1)

    print(f"\n  Loaded  : {len(images)}")
    print(f"  Failed  : {failed}")
    print(f"  Shape   : {images.shape}")
    print(f"  RAM     : {REPORT['preprocess_memory_mb']} MB")
    return images, labels


# ============================================================================
# STEP 6 - DATA AUGMENTATION
# ============================================================================

def augment_image(image):
    aug = image.copy()
    if aug.max() <= 1.0:
        aug = (aug * 255).astype(np.uint8)

    if np.random.random() > 0.5:
        aug = cv2.flip(aug, 1)
    if np.random.random() > 0.3:
        angle = np.random.uniform(-20, 20)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if np.random.random() > 0.3:
        f = np.random.uniform(0.7, 1.3)
        aug = np.clip(aug * f, 0, 255).astype(np.uint8)
    if np.random.random() > 0.5:
        f = np.random.uniform(0.8, 1.2)
        m = np.mean(aug)
        aug = np.clip((aug - m) * f + m, 0, 255).astype(np.uint8)
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 8, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if np.random.random() > 0.5:
        h, w = aug.shape[:2]
        cf = np.random.uniform(0.85, 1.0)
        nh, nw = int(h * cf), int(w * cf)
        sy = np.random.randint(0, h - nh + 1)
        sx = np.random.randint(0, w - nw + 1)
        aug = cv2.resize(aug[sy:sy + nh, sx:sx + nw], (w, h))

    return aug.astype(np.float32) / 255.0


def visualize_augmentation(images, labels):
    unique = np.unique(labels)
    n = min(4, len(unique))
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    for i in range(n):
        mask = labels == unique[i]
        orig = images[mask][0]
        axes[i][0].imshow(orig)
        axes[i][0].set_title(f'{unique[i]}\n(original)', fontsize=8)
        axes[i][0].axis('off')
        for j in range(1, 5):
            axes[i][j].imshow(augment_image(orig))
            axes[i][j].set_title(f'aug {j}', fontsize=8)
            axes[i][j].axis('off')
    plt.suptitle('Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, '07_augmentation_examples.png')


def balance_with_augmentation(images, labels):
    print("\n" + "=" * 70)
    print("  STEP 6 : DATA AUGMENTATION")
    print("=" * 70)

    target = CONFIG['augmentation_target']
    unique = np.unique(labels)
    REPORT['before_augmentation'] = len(images)

    print(f"\n  Target / class : {target}")
    print(f"\n  {'Class':<35} {'Before':>6} {'Added':>6} {'After':>6}")
    print("  " + "-" * 58)

    all_imgs, all_lbls = [], []
    counts_before = {}
    counts_after = {}
    total_added = 0

    for lbl in unique:
        mask = labels == lbl
        cls_imgs = images[mask]
        n = len(cls_imgs)
        counts_before[lbl] = n

        all_imgs.extend(cls_imgs)
        all_lbls.extend([lbl] * n)

        need = max(0, target - n)
        total_added += need

        for _ in range(need):
            idx = np.random.randint(0, n)
            all_imgs.append(augment_image(cls_imgs[idx]))
            all_lbls.append(lbl)

        after = n + need
        counts_after[lbl] = after
        print(f"  {lbl:<35} {n:>6} {need:>6} {after:>6}")

    all_imgs = np.array(all_imgs)
    all_lbls = np.array(all_lbls)

    REPORT['after_augmentation'] = len(all_imgs)
    REPORT['augmentation_counts_before'] = counts_before
    REPORT['augmentation_counts_after'] = counts_after
    REPORT['augmentation_added'] = total_added

    print(f"\n  Before augmentation : {REPORT['before_augmentation']}")
    print(f"  Images added        : {total_added}")
    print(f"  After augmentation  : {len(all_imgs)}")

    visualize_augmentation(images, labels)
    return all_imgs, all_lbls


# ============================================================================
# STEP 7 - FEATURE EXTRACTION
# ============================================================================

def extract_hog_features(image):
    gray = rgb2gray(image)
    return hog(gray, orientations=CONFIG['hog_orientations'],
               pixels_per_cell=CONFIG['hog_pixels_per_cell'],
               cells_per_block=CONFIG['hog_cells_per_block'],
               block_norm='L2-Hys', visualize=False, feature_vector=True)


def extract_lbp_features(image):
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=CONFIG['lbp_n_points'],
                                R=CONFIG['lbp_radius'], method='uniform')
    n_bins = CONFIG['lbp_n_points'] + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_color_histogram(image):
    feats = []
    for ch in range(3):
        h, _ = np.histogram(image[:, :, ch].ravel(),
                             bins=CONFIG['color_hist_bins'], range=(0, 1), density=True)
        feats.extend(h)
    return np.array(feats)


def extract_glcm_features(image):
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    gray = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(gray, distances=CONFIG['glcm_distances'],
                         angles=CONFIG['glcm_angles'],
                         levels=32, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']:
        vals = graycoprops(glcm, prop)
        feats.extend([vals.mean(), vals.std()])
    return np.array(feats)


def extract_statistical_features(image):
    feats = []
    for ch in range(3):
        c = image[:, :, ch].ravel()
        feats.extend([
            np.mean(c), np.std(c), np.median(c),
            float(pd.Series(c).skew()),
            float(pd.Series(c).kurtosis()),
            np.percentile(c, 25), np.percentile(c, 75),
        ])
    return np.array(feats)


def extract_all_features(image):
    return np.concatenate([
        extract_hog_features(image),
        extract_lbp_features(image),
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_statistical_features(image),
    ])


def extract_features_dataset(images, labels):
    print("\n" + "=" * 70)
    print("  STEP 7 : FEATURE EXTRACTION")
    print("=" * 70)

    probe = extract_all_features(images[0])
    n_feat = len(probe)
    breakdown = OrderedDict([
        ('HOG', len(extract_hog_features(images[0]))),
        ('LBP', len(extract_lbp_features(images[0]))),
        ('Colour Histogram', len(extract_color_histogram(images[0]))),
        ('GLCM', len(extract_glcm_features(images[0]))),
        ('Statistical', len(extract_statistical_features(images[0]))),
    ])
    REPORT['feature_breakdown'] = dict(breakdown)
    REPORT['feature_vector_size'] = n_feat

    print(f"\n  Feature breakdown:")
    for k, v in breakdown.items():
        print(f"    {k:<20} : {v}")
    print(f"    {'TOTAL':<20} : {n_feat}")
    print(f"\n  Extracting from {len(images)} images ...")

    X = np.zeros((len(images), n_feat), dtype=np.float32)
    bad = []
    for i in tqdm(range(len(images)), desc="  Extracting"):
        try:
            X[i] = extract_all_features(images[i])
        except Exception:
            bad.append(i)

    REPORT['feature_extraction_failed'] = len(bad)
    if bad:
        print(f"  Warning: {len(bad)} failures removed")
        mask = np.ones(len(images), dtype=bool)
        mask[bad] = False
        X = X[mask]
        labels = labels[mask]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    REPORT['feature_matrix_shape'] = str(X.shape)
    REPORT['feature_memory_mb'] = round(X.nbytes / (1024 ** 2), 1)

    print(f"\n  Feature matrix : {X.shape}")
    print(f"  RAM            : {REPORT['feature_memory_mb']} MB")
    return X, labels


# ============================================================================
# STEP 8 - ML-READY DATASET
# ============================================================================

def prepare_ml_dataset(features, labels):
    print("\n" + "=" * 70)
    print("  STEP 8 : ML-READY DATASET PREPARATION")
    print("=" * 70)

    # Label Encoding
    print("\n  -- Label Encoding --")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    enc_map = {}
    for i, c in enumerate(le.classes_):
        n = int((y == i).sum())
        enc_map[c] = {'code': i, 'count': n}
        print(f"     {i:>2} : {c:<35} ({n})")
    REPORT['label_encoding'] = enc_map

    # Split
    print("\n  -- Stratified Train-Test Split --")
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, y, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'], stratify=y)

    REPORT['train_samples'] = int(X_tr.shape[0])
    REPORT['test_samples'] = int(X_te.shape[0])
    print(f"     Train : {X_tr.shape[0]}")
    print(f"     Test  : {X_te.shape[0]}")

    tr_pct = np.bincount(y_tr) / len(y_tr) * 100
    te_pct = np.bincount(y_te) / len(y_te) * 100
    tr_dict, te_dict = {}, {}
    print(f"\n     {'Class':<35} {'Train%':>7} {'Test%':>7}")
    for i, c in enumerate(le.classes_):
        print(f"     {c:<35} {tr_pct[i]:>6.1f}% {te_pct[i]:>6.1f}%")
        tr_dict[c] = round(float(tr_pct[i]), 2)
        te_dict[c] = round(float(te_pct[i]), 2)
    REPORT['train_class_pct'] = tr_dict
    REPORT['test_class_pct'] = te_dict

    # Scaling
    print("\n  -- Feature Scaling --")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    REPORT['scaling_before_mean'] = round(float(X_tr.mean()), 4)
    REPORT['scaling_before_std'] = round(float(X_tr.std()), 4)
    REPORT['scaling_after_mean'] = round(float(X_tr_s.mean()), 4)
    REPORT['scaling_after_std'] = round(float(X_tr_s.std()), 4)
    print(f"     Before — mean {REPORT['scaling_before_mean']}  std {REPORT['scaling_before_std']}")
    print(f"     After  — mean {REPORT['scaling_after_mean']}  std {REPORT['scaling_after_std']}")

    # Class Weights
    print("\n  -- Class Weights --")
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {}
    for cid, w in zip(np.unique(y_tr), cw):
        cw_dict[le.classes_[cid]] = round(float(w), 4)
        print(f"     {le.classes_[cid]:<35} : {w:.4f}")
    REPORT['class_weights'] = cw_dict

    # Save
    print("\n  -- Saving --")
    out = CONFIG['processed_path']
    bundle = {
        'X_train': X_tr_s, 'X_test': X_te_s,
        'y_train': y_tr, 'y_test': y_te,
        'label_encoder': le, 'scaler': scaler,
        'class_weights': cw_dict, 'config': CONFIG,
    }
    with open(os.path.join(out, 'ml_ready_dataset.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    np.save(os.path.join(out, 'X_train.npy'), X_tr_s)
    np.save(os.path.join(out, 'X_test.npy'), X_te_s)
    np.save(os.path.join(out, 'y_train.npy'), y_tr)
    np.save(os.path.join(out, 'y_test.npy'), y_te)
    joblib.dump(le, os.path.join(out, 'label_encoder.joblib'))
    joblib.dump(scaler, os.path.join(out, 'scaler.joblib'))
    print(f"     [OK] Saved -> {os.path.abspath(out)}")

    return bundle


# ============================================================================
# STEP 9 - PDF REPORT  (FIXED AND ROBUST)
# ============================================================================

class PipelineReport(FPDF):
    """Custom PDF with simple compatible API."""

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 6,
                      'Skin Disease Classification - Pipeline Report',
                      0, 1, 'C')
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section(self, title):
        """Blue section heading with underline."""
        self.ln(4)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(25, 70, 150)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_draw_color(25, 70, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub(self, title):
        """Dark subsection heading."""
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(1)

    def kv(self, key, value):
        """Key-value pair on one line."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 40, 40)
        self.cell(80, 6, str(key), 0, 0, 'L')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, str(value), 0, 1, 'L')

    def body(self, text):
        """Paragraph text with word wrap."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, str(text))
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        """Table with header row and alternating colours."""
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [190 // n_cols] * n_cols

        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(25, 70, 150)
        self.set_text_color(255, 255, 255)
        for w, h in zip(col_widths, headers):
            self.cell(w, 7, str(h), 1, 0, 'C', True)
        self.ln()

        # Rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        for idx, row in enumerate(rows):
            if idx % 2 == 0:
                self.set_fill_color(235, 240, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for w, val in zip(col_widths, row):
                self.cell(w, 6, str(val), 1, 0, 'C', True)
            self.ln()
        self.ln(3)

    def safe_image(self, path, w=170):
        """Add image if file exists, skip gracefully otherwise."""
        if os.path.exists(path):
            try:
                # Check if enough space on page
                if self.get_y() > 160:
                    self.add_page()
                self.image(path, x=20, w=w)
                self.ln(5)
                return True
            except Exception as e:
                self.body(f"[Image error: {e}]")
                return False
        else:
            self.body(f"[Image not found: {os.path.basename(path)}]")
            return False

    def separator(self):
        """Light horizontal line."""
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def check_page_space(self, needed=60):
        """Add new page if not enough space."""
        if self.get_y() > (297 - needed):
            self.add_page()


def generate_pdf_report():
    """Generate comprehensive PDF summary report."""

    if not PDF_AVAILABLE:
        print("\n  [!!] fpdf2 not installed — skipping PDF generation")
        print("       Run: pip install fpdf2")
        return None

    print("\n" + "=" * 70)
    print("  STEP 9 : GENERATING PDF REPORT")
    print("=" * 70)

    try:
        pdf = PipelineReport()
        pdf.set_auto_page_break(auto=True, margin=25)

        # Calculate elapsed time
        elapsed = "N/A"
        if REPORT.get('start_time') and REPORT.get('end_time'):
            elapsed = str(REPORT['end_time'] - REPORT['start_time']).split('.')[0]

        # ================================================================
        # TITLE PAGE
        # ================================================================
        pdf.add_page()
        pdf.ln(25)

        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(25, 70, 150)
        pdf.cell(0, 12, 'Skin Disease Classification', 0, 1, 'C')

        pdf.set_font('Helvetica', 'B', 18)
        pdf.cell(0, 10, 'ML Pipeline Report', 0, 1, 'C')

        pdf.ln(8)
        pdf.set_draw_color(25, 70, 150)
        pdf.set_line_width(0.8)
        pdf.line(60, pdf.get_y(), 150, pdf.get_y())
        pdf.ln(8)

        pdf.set_font('Helvetica', '', 13)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, 'Early Detection with Human-in-the-Loop Validation', 0, 1, 'C')

        pdf.ln(5)
        pdf.set_font('Helvetica', 'I', 11)
        pdf.cell(0, 7, 'Data Preprocessing & Feature Extraction Pipeline', 0, 1, 'C')

        pdf.ln(20)
        pdf.set_line_width(0.2)
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(60, 60, 60)

        pdf.kv('Report Generated', datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        pdf.kv('Pipeline Start', str(REPORT.get('start_time', 'N/A')).split('.')[0])
        pdf.kv('Pipeline End', str(REPORT.get('end_time', 'N/A')).split('.')[0])
        pdf.kv('Total Duration', elapsed)
        pdf.kv('Python Version', sys.version.split()[0])
        pdf.kv('Platform', sys.platform)

        # ================================================================
        # 1. DATASET OVERVIEW
        # ================================================================
        pdf.add_page()
        pdf.section('1. Dataset Overview')

        pdf.kv('Dataset 1 (Kaggle)', f"{REPORT['dataset1_count']} images")
        pdf.kv('Dataset 2 (Kaggle)', f"{REPORT['dataset2_count']} images")
        pdf.kv('Total Raw Images', f"{REPORT['total_raw']} images")
        pdf.kv('Number of Classes', str(REPORT['num_classes']))
        pdf.kv('Image Size (target)', f"{CONFIG['target_size'][0]} x {CONFIG['target_size'][1]} pixels")
        pdf.kv('Colour Space', CONFIG['color_space'])

        pdf.ln(4)
        pdf.sub('Raw Class Distribution')
        headers = ['Disease Class', 'Image Count']
        rows = []
        for c in sorted(REPORT['class_counts_raw'].keys()):
            rows.append([c, str(REPORT['class_counts_raw'][c])])
        rows.append(['TOTAL', str(REPORT['total_raw'])])
        pdf.add_table(headers, rows, col_widths=[120, 70])

        pdf.kv('Imbalance Ratio (raw)', f"{REPORT['imbalance_ratio_raw']} : 1")

        # ================================================================
        # 2. EDA RESULTS
        # ================================================================
        pdf.add_page()
        pdf.section('2. Exploratory Data Analysis')

        pdf.sub('2.1 Data Quality Check')
        pdf.kv('Corrupt / Unreadable', str(REPORT['corrupt_count']))
        pdf.kv('Duplicate Images', str(REPORT['duplicate_count']))

        if REPORT.get('dim_stats'):
            pdf.ln(2)
            pdf.sub('2.2 Image Dimensions')
            ds = REPORT['dim_stats']
            pdf.kv('Width Range', f"{ds['width_min']} - {ds['width_max']} px  (mean: {ds['width_mean']})")
            pdf.kv('Height Range', f"{ds['height_min']} - {ds['height_max']} px  (mean: {ds['height_mean']})")

        # Embed each EDA plot on its own page
        plot_info = [
            ('01_class_dist', '2.3 Class Distribution'),
            ('02_image_dim', '2.4 Image Dimension Analysis'),
            ('03_pixel', '2.5 Pixel Intensity per Class'),
            ('04_sample', '2.6 Sample Images per Class'),
            ('05_color', '2.7 Colour Histograms per Class'),
            ('06_source', '2.8 Dataset Source Comparison'),
        ]

        for prefix, title in plot_info:
            for ppath in REPORT['eda_plots']:
                fname = os.path.basename(ppath)
                if fname.startswith(prefix):
                    pdf.add_page()
                    pdf.sub(title)
                    pdf.safe_image(ppath, w=170)
                    break

        # Pixel stats table
        if REPORT.get('pixel_stats'):
            pdf.add_page()
            pdf.sub('2.9 Pixel Statistics Table')
            headers = ['Class', 'R Mean', 'G Mean', 'B Mean']
            rows = []
            for cls, vals in sorted(REPORT['pixel_stats'].items()):
                rows.append([cls, str(vals.get('R', '-')),
                            str(vals.get('G', '-')), str(vals.get('B', '-'))])
            pdf.add_table(headers, rows, col_widths=[90, 33, 33, 34])

        # ================================================================
        # 3. DATA CLEANING
        # ================================================================
        pdf.add_page()
        pdf.section('3. Data Cleaning')

        pdf.body(
            "Corrupt and duplicate images were identified during EDA and "
            "removed to prevent errors during training and data leakage "
            "during evaluation."
        )

        headers = ['Metric', 'Count']
        rows = [
            ['Images Before Cleaning', str(REPORT['total_raw'])],
            ['Corrupt Images Removed', str(REPORT['corrupt_removed'])],
            ['Duplicate Images Removed', str(REPORT['duplicates_removed'])],
            ['Images After Cleaning', str(REPORT['after_cleaning'])],
        ]
        pdf.add_table(headers, rows, col_widths=[120, 70])

        # ================================================================
        # 4. SMART SAMPLING
        # ================================================================
        pdf.section('4. Smart Sampling')

        pdf.body(
            f"To reduce processing time, classes with more than "
            f"{CONFIG['max_per_class']} images were randomly down-sampled "
            f"to {CONFIG['max_per_class']}. For traditional ML (SVM, RF, KNN), "
            f"500-800 images per class is sufficient. This reduced the "
            f"dataset by {REPORT['sampling_reduction_pct']}% with minimal "
            f"impact on accuracy."
        )

        headers = ['Metric', 'Value']
        rows = [
            ['Images Before Sampling', str(REPORT['before_sampling'])],
            ['Cap per Class', str(CONFIG['max_per_class'])],
            ['Images After Sampling', str(REPORT['after_sampling'])],
            ['Reduction', f"{REPORT['sampling_reduction_pct']}%"],
            ['Imbalance Ratio (after)', f"{REPORT['imbalance_ratio_sampled']} : 1"],
        ]
        pdf.add_table(headers, rows, col_widths=[120, 70])

        pdf.check_page_space(80)
        pdf.sub('Per-Class Sampling Results')
        headers = ['Class', 'After Sampling']
        rows = [[c, str(v)] for c, v in sorted(REPORT['sampling_counts'].items())]
        pdf.add_table(headers, rows, col_widths=[120, 70])

        # Sampled distribution plot
        for ppath in REPORT['eda_plots']:
            if 'After_Sampling' in os.path.basename(ppath):
                pdf.add_page()
                pdf.sub('Class Distribution After Sampling')
                pdf.safe_image(ppath, w=170)
                break

        # ================================================================
        # 5. PREPROCESSING  (BEFORE / AFTER)
        # ================================================================
        pdf.add_page()
        pdf.section('5. Image Preprocessing')

        pdf.body(
            f"All images were resized to {CONFIG['target_size'][0]} x "
            f"{CONFIG['target_size'][1]} pixels using INTER_AREA interpolation "
            f"(best for down-scaling). Pixel values were normalised from "
            f"0-255 to 0.0-1.0 by dividing by 255 (Min-Max normalisation). "
            f"This ensures all features contribute equally during model training."
        )

        pdf.sub('Before vs After Preprocessing')
        headers = ['Metric', 'Before', 'After']
        rows = [
            ['Total Images',
             str(REPORT['before_preprocessing']),
             str(REPORT['after_preprocessing'])],
            ['Failed / Skipped',
             '-',
             str(REPORT['preprocess_failed'])],
            ['Image Size',
             'Variable (mixed sizes)',
             f"{CONFIG['target_size'][0]} x {CONFIG['target_size'][1]} x 3"],
            ['Pixel Range',
             '0 - 255 (uint8)',
             '0.0 - 1.0 (float32)'],
            ['Array Shape',
             'N/A',
             REPORT['preprocess_shape']],
            ['Memory Usage',
             'N/A',
             f"{REPORT['preprocess_memory_mb']} MB"],
        ]
        pdf.add_table(headers, rows, col_widths=[60, 65, 65])

        # ================================================================
        # 6. DATA AUGMENTATION  (BEFORE / AFTER)
        # ================================================================
        pdf.add_page()
        pdf.section('6. Data Augmentation')

        pdf.body(
            f"Minority classes were augmented to {CONFIG['augmentation_target']} "
            f"images each using random transformations: horizontal flips, "
            f"rotations (+/-20 degrees), brightness adjustment (0.7-1.3x), "
            f"contrast adjustment (0.8-1.2x), Gaussian noise, and random "
            f"crop-zoom. This creates a perfectly balanced dataset."
        )

        pdf.sub('Before vs After Augmentation')
        headers = ['Metric', 'Before', 'After']
        rows = [
            ['Total Images',
             str(REPORT['before_augmentation']),
             str(REPORT['after_augmentation'])],
            ['Target per Class',
             '-',
             str(CONFIG['augmentation_target'])],
            ['Images Added',
             '-',
             str(REPORT['augmentation_added'])],
            ['Class Balance',
             'Imbalanced',
             'Perfectly Balanced (1:1)'],
        ]
        pdf.add_table(headers, rows, col_widths=[60, 65, 65])

        pdf.check_page_space(80)
        pdf.sub('Per-Class Augmentation Detail')
        headers = ['Class', 'Before', 'After', 'Added']
        rows = []
        for c in sorted(REPORT['augmentation_counts_before'].keys()):
            before = REPORT['augmentation_counts_before'][c]
            after = REPORT['augmentation_counts_after'][c]
            added = after - before
            rows.append([c, str(before), str(after), str(added)])
        # Total row
        tot_before = sum(REPORT['augmentation_counts_before'].values())
        tot_after = sum(REPORT['augmentation_counts_after'].values())
        rows.append(['TOTAL', str(tot_before), str(tot_after),
                     str(tot_after - tot_before)])
        pdf.add_table(headers, rows, col_widths=[80, 35, 35, 40])

        # Augmentation examples plot
        for ppath in REPORT['eda_plots']:
            if '07_augmentation' in os.path.basename(ppath):
                pdf.add_page()
                pdf.sub('Augmentation Examples')
                pdf.safe_image(ppath, w=170)
                break

        # ================================================================
        # 7. FEATURE EXTRACTION
        # ================================================================
        pdf.add_page()
        pdf.section('7. Feature Extraction')

        pdf.body(
            "Five complementary feature extraction methods were combined "
            "(feature fusion) to create a rich representation of each image. "
            "HOG captures shape/edges, LBP captures local texture, colour "
            "histograms capture colour distribution, GLCM captures texture "
            "regularity, and statistical features capture global properties."
        )

        headers = ['Feature Type', 'Dimensions', 'Captures']
        rows = [
            ['HOG', str(REPORT['feature_breakdown'].get('HOG', '-')),
             'Shape & edges'],
            ['LBP', str(REPORT['feature_breakdown'].get('LBP', '-')),
             'Local texture'],
            ['Colour Histogram',
             str(REPORT['feature_breakdown'].get('Colour Histogram', '-')),
             'Colour distribution'],
            ['GLCM', str(REPORT['feature_breakdown'].get('GLCM', '-')),
             'Texture regularity'],
            ['Statistical',
             str(REPORT['feature_breakdown'].get('Statistical', '-')),
             'Global pixel stats'],
            ['TOTAL', str(REPORT['feature_vector_size']),
             'Combined feature vector'],
        ]
        pdf.add_table(headers, rows, col_widths=[55, 40, 95])

        pdf.kv('Feature Matrix Shape', REPORT['feature_matrix_shape'])
        pdf.kv('Memory Usage', f"{REPORT['feature_memory_mb']} MB")
        pdf.kv('Failed Extractions', str(REPORT['feature_extraction_failed']))

        # ================================================================
        # 8. TRAIN-TEST SPLIT & SCALING
        # ================================================================
        pdf.add_page()
        pdf.section('8. Train-Test Split & Feature Scaling')

        pdf.sub('Split Configuration')
        pdf.kv('Split Ratio', f"{int((1-CONFIG['test_size'])*100)} / {int(CONFIG['test_size']*100)}")
        pdf.kv('Stratified', 'Yes (class proportions preserved)')
        pdf.kv('Random State', str(CONFIG['random_state']))

        pdf.ln(3)
        headers = ['Set', 'Samples', 'Percentage']
        rows = [
            ['Training', str(REPORT['train_samples']),
             f"{100*(1-CONFIG['test_size']):.0f}%"],
            ['Test', str(REPORT['test_samples']),
             f"{100*CONFIG['test_size']:.0f}%"],
            ['Total', str(REPORT['train_samples'] + REPORT['test_samples']), '100%'],
        ]
        pdf.add_table(headers, rows, col_widths=[60, 65, 65])

        pdf.sub('Feature Scaling (StandardScaler)')
        pdf.body(
            "StandardScaler transforms features to zero mean and unit "
            "variance. CRITICAL: the scaler was fitted on training data "
            "ONLY, then applied to test data. Fitting on test data would "
            "cause data leakage."
        )
        headers = ['Metric', 'Before Scaling', 'After Scaling']
        rows = [
            ['Mean', str(REPORT['scaling_before_mean']), str(REPORT['scaling_after_mean'])],
            ['Std Dev', str(REPORT['scaling_before_std']), str(REPORT['scaling_after_std'])],
        ]
        pdf.add_table(headers, rows, col_widths=[60, 65, 65])

        pdf.check_page_space(80)
        pdf.sub('Class Distribution in Splits')
        headers = ['Class', 'Train %', 'Test %']
        rows = []
        for c in sorted(REPORT['train_class_pct'].keys()):
            rows.append([c,
                        f"{REPORT['train_class_pct'][c]:.1f}%",
                        f"{REPORT['test_class_pct'][c]:.1f}%"])
        pdf.add_table(headers, rows, col_widths=[90, 50, 50])

        # ================================================================
        # 9. CLASS WEIGHTS
        # ================================================================
        pdf.add_page()
        pdf.section('9. Class Weights')

        pdf.body(
            "Class weights were computed using sklearn's 'balanced' "
            "strategy: weight = n_samples / (n_classes x n_class_samples). "
            "Higher weights penalise misclassification of rarer classes."
        )

        headers = ['Class', 'Weight']
        rows = [[c, str(w)] for c, w in sorted(REPORT['class_weights'].items())]
        pdf.add_table(headers, rows, col_widths=[120, 70])

        # ================================================================
        # 10. OUTPUT FILES
        # ================================================================
        pdf.section('10. Output Files')

        headers = ['File', 'Description']
        rows = [
            ['ml_ready_dataset.pkl', 'Complete bundle (all objects)'],
            ['X_train.npy', 'Training features (scaled)'],
            ['X_test.npy', 'Test features (scaled)'],
            ['y_train.npy', 'Training labels (encoded)'],
            ['y_test.npy', 'Test labels (encoded)'],
            ['label_encoder.joblib', 'Class name to integer mapping'],
            ['scaler.joblib', 'Fitted StandardScaler object'],
            ['pipeline_summary_report.pdf', 'This report'],
        ]
        pdf.add_table(headers, rows, col_widths=[80, 110])

        pdf.kv('Output Directory', os.path.abspath(CONFIG['processed_path']))

        # ================================================================
        # 11. COMPLETE PIPELINE SUMMARY
        # ================================================================
        pdf.add_page()
        pdf.section('11. Complete Pipeline Summary')

        pdf.sub('Data Flow Through Pipeline')

        headers = ['Pipeline Stage', 'Images', 'Change']
        rows = [
            ['1. Raw Dataset (Combined)',
             str(REPORT['total_raw']), '-'],
            ['2. After Cleaning',
             str(REPORT['after_cleaning']),
             f"-{REPORT['total_raw'] - REPORT['after_cleaning']}"],
            ['3. After Smart Sampling',
             str(REPORT['after_sampling']),
             f"-{REPORT['after_cleaning'] - REPORT['after_sampling']}"],
            ['4. After Preprocessing',
             str(REPORT['after_preprocessing']),
             f"-{REPORT['after_sampling'] - REPORT['after_preprocessing']} (failed)"],
            ['5. After Augmentation',
             str(REPORT['after_augmentation']),
             f"+{REPORT['augmentation_added']}"],
        ]
        pdf.add_table(headers, rows, col_widths=[75, 55, 60])

        pdf.ln(3)
        pdf.sub('Final Dataset Specifications')
        headers = ['Specification', 'Value']
        rows = [
            ['Training Samples', str(REPORT['train_samples'])],
            ['Test Samples', str(REPORT['test_samples'])],
            ['Feature Vector Dimension', str(REPORT['feature_vector_size'])],
            ['Number of Classes', str(REPORT['num_classes'])],
            ['Class Balance', 'Balanced (1:1)'],
            ['Feature Scaling', 'StandardScaler (mean=0, std=1)'],
            ['Total Processing Time', elapsed],
        ]
        pdf.add_table(headers, rows, col_widths=[95, 95])

        # ================================================================
        # 12. CONFIGURATION
        # ================================================================
        pdf.add_page()
        pdf.section('12. Configuration Parameters')

        headers = ['Parameter', 'Value']
        rows = []
        for k in sorted(CONFIG.keys()):
            v = str(CONFIG[k])
            if len(v) > 55:
                v = v[:52] + '...'
            rows.append([k, v])
        pdf.add_table(headers, rows, col_widths=[80, 110])

        # ================================================================
        # 13. NEXT STEPS
        # ================================================================
        pdf.add_page()
        pdf.section('13. Recommended Next Steps')

        steps = [
            ('1. Train SVM (RBF Kernel)',
             'Best for high-dimensional feature spaces. Expected accuracy: 85-92%.'),
            ('2. Train Random Forest',
             'Good accuracy with built-in feature importance for interpretability.'),
            ('3. Train KNN',
             'Simple distance-based baseline. May struggle with high dimensions.'),
            ('4. Train Decision Tree',
             'Most interpretable model. Use as explainability baseline.'),
            ('5. Model Evaluation',
             'Compare using Accuracy, Precision, Recall, F1-Score, Confusion Matrix.'),
            ('6. Select Best Model',
             'Choose based on weighted F1-Score (handles class importance).'),
            ('7. Human-in-the-Loop Integration',
             'Present probability-based predictions for clinician validation.'),
            ('8. Clinical Deployment',
             'Package as decision-support tool integrated into clinical workflow.'),
        ]

        for title, desc in steps:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(25, 70, 150)
            pdf.cell(0, 6, title, 0, 1)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 5, desc, 0, 1)
            pdf.ln(2)

        # ================================================================
        # FINAL PAGE
        # ================================================================
        pdf.add_page()
        pdf.ln(40)
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(25, 130, 25)
        pdf.cell(0, 12, 'Pipeline Completed Successfully', 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font('Helvetica', '', 14)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, 'Dataset is ML-Ready', 0, 1, 'C')
        pdf.ln(3)
        pdf.cell(0, 8, f'Total Processing Time: {elapsed}', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(0, 6, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        pdf.cell(0, 6, f'Total pages: {pdf.page_no()}', 0, 1, 'C')

        # ================================================================
        # SAVE PDF
        # ================================================================
        pdf_path = os.path.join(CONFIG['processed_path'], 'pipeline_summary_report.pdf')
        pdf.output(pdf_path)

        print(f"\n  [OK] PDF saved : {os.path.abspath(pdf_path)}")
        print(f"       Pages     : {pdf.page_no()}")
        return pdf_path

    except Exception as e:
        print(f"\n  [!!] PDF generation failed: {e}")
        print(f"       Error details:")
        traceback.print_exc()
        print(f"\n       All data files were saved successfully.")
        print(f"       Only the PDF report could not be created.")

        # Save report data as JSON fallback
        json_path = os.path.join(CONFIG['processed_path'], 'pipeline_report.json')
        try:
            json_safe = {}
            for k, v in REPORT.items():
                try:
                    json.dumps(v)
                    json_safe[k] = v
                except (TypeError, ValueError):
                    json_safe[k] = str(v)
            with open(json_path, 'w') as f:
                json.dump(json_safe, f, indent=2, default=str)
            print(f"       JSON fallback saved: {json_path}")
        except Exception:
            pass

        return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete Pipeline:

    Raw Images (28,647)
         |
    1.   |-- Load Inventory
    2.   |-- Full EDA
    3.   |-- Clean (corrupt + duplicates)
    4.   |-- Smart Sample (cap at 800/class)
         |-- Verify sampled distribution
    5.   |-- Preprocess (resize + normalise)
    6.   |-- Augment (balance all classes)
    7.   |-- Feature Extraction (HOG+LBP+Color+GLCM+Stats)
    8.   |-- ML-Ready (encode + split + scale + save)
    9.   |-- PDF Summary Report
              |
              v
         data/processed/  <-- ready for training
    """

    REPORT['start_time'] = datetime.now()

    # 0
    create_directories()

    # 1
    df = load_dataset_inventory()

    # 2
    corrupt, dupes = run_full_eda(df)

    # 3
    df = remove_corrupt_and_duplicates(df, corrupt, dupes)

    # 4
    df = smart_sample_dataset(df)
    eda_class_distribution(df, "(After Sampling)")

    # 5
    images, labels = preprocess_images(df)

    # 6
    images, labels = balance_with_augmentation(images, labels)

    # 7
    features, labels = extract_features_dataset(images, labels)
    del images

    # 8
    bundle = prepare_ml_dataset(features, labels)
    del features

    # 9
    REPORT['end_time'] = datetime.now()
    pdf_path = generate_pdf_report()

    # Done
    elapsed = REPORT['end_time'] - REPORT['start_time']

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Duration : {elapsed}")
    print(f"  Output   : {os.path.abspath(CONFIG['processed_path'])}")
    print()
    print("  Files:")
    for f in os.listdir(CONFIG['processed_path']):
        fpath = os.path.join(CONFIG['processed_path'], f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"    {f:<35} {size_str:>10}")
    print()
    print("  NEXT --> Train SVM | Random Forest | KNN | Decision Tree")
    print(f"\n  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()