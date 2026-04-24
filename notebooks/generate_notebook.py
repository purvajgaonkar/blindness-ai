import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Retinal Image Preprocessing & Analysis Pipeline\nThis notebook demonstrates classical computer vision techniques applied to retinal fundus images for Diabetic Retinopathy detection, as part of the APTOS 2019 dataset analysis. It follows the exact pipeline and techniques specified."),
    nbf.v4.new_code_cell("""import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from glob import glob
import os
import random
from collections import Counter
from sklearn.manifold import TSNE
from skimage.feature import local_binary_pattern
from skimage.filters import frangi, sato

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 6)

DATA_PATH = r"D:/VIT/Projects/blindness-ai/aptos2019_data"
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train_images")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")

train_df = pd.read_csv(TRAIN_CSV)
print("Total training images:", len(train_df))"""),

    # SECTION 1
    nbf.v4.new_markdown_cell("## SECTION 1 — IMAGE LOADING & BASIC STATS\n### 1. Load 5 sample images (one from each DR class)\nLoads one representative image from each severity grade (0 to 4)."),
    nbf.v4.new_code_cell("""# 1. Load 5 sample images
def get_sample_images(df, img_dir, num_classes=5):
    samples = []
    for c in range(num_classes):
        img_id = df[df['diagnosis'] == c].sample(1, random_state=42)['id_code'].values[0]
        img_path = os.path.join(img_dir, f"{img_id}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        samples.append((img, c, img_id))
    return samples

samples = get_sample_images(train_df, TRAIN_IMG_DIR)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
for ax, (img, label, img_id) in zip(axes, samples):
    ax.imshow(img)
    ax.set_title(f"Class {label}: {classes[label]}", fontsize=12, color='cyan')
    ax.axis('off')
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("### 2. Image Histograms\nHistograms show the distribution of pixel intensities for Red, Green, and Blue channels."),
    nbf.v4.new_code_cell("""# 2. Show image histograms (R, G, B channels)
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
colors = ('r', 'g', 'b')
for ax, (img, label, _) in zip(axes, samples):
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, alpha=0.7)
    ax.set_title(f"Histogram - Class {label}")
    ax.set_xlim([0, 256])
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("### 3. Display Image Metadata\nShape, data type, min/max values, and mean intensity."),
    nbf.v4.new_code_cell("""# 3. Display image metadata
for img, label, img_id in samples:
    print(f"[{classes[label]}] ID: {img_id}")
    print(f"  Shape: {img.shape}, Dtype: {img.dtype}")
    print(f"  Min/Max: {img.min()}/{img.max()}")
    mean_color = img.mean(axis=(0, 1))
    print(f"  Mean Intensity -> R: {mean_color[0]:.1f}, G: {mean_color[1]:.1f}, B: {mean_color[2]:.1f}\\n")"""),

    # SECTION 2
    nbf.v4.new_markdown_cell("## SECTION 2 — COLOR SPACE TRANSFORMATIONS\nVarious color spaces extract different types of information. Green channel generally contains the most contrast for retinal structures like blood vessels and lesions."),
    nbf.v4.new_code_cell("""sample_img = cv2.resize(samples[2][0], (512, 512)) # Use a Moderate DR image for demonstrations

# 4. RGB to Grayscale
gray_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sample_img)
axes[0].set_title("Original RGB")
axes[0].axis('off')
axes[1].imshow(gray_img, cmap='gray')
axes[1].set_title("Grayscale")
axes[1].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 5. RGB to HSV"),
    nbf.v4.new_code_cell("""# 5. RGB to HSV conversion
hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsv_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(H, cmap='hsv'); axes[0].set_title("Hue (H)")
axes[1].imshow(S, cmap='gray'); axes[1].set_title("Saturation (S)")
axes[2].imshow(V, cmap='gray'); axes[2].set_title("Value (V)")
for ax in axes: ax.axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 6. RGB to LAB"),
    nbf.v4.new_code_cell("""# 6. RGB to LAB color space
lab_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(lab_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(L, cmap='gray'); axes[0].set_title("Luminance (L)")
axes[1].imshow(A, cmap='gray'); axes[1].set_title("A (Green-Red)")
axes[2].imshow(B, cmap='gray'); axes[2].set_title("B (Blue-Yellow)")
for ax in axes: ax.axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 7. Extract the GREEN CHANNEL\nThe green channel is critical in retinal imaging because the red channel is overly saturated (redness of the fundus) and the blue channel suffers from low contrast and scattering. Green provides the highest contrast for blood vessels, exudates, and hemorrhages."),
    nbf.v4.new_code_cell("""# 7. Extract GREEN CHANNEL
R, G, B_ch = cv2.split(sample_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(R, cmap='Reds'); axes[0].set_title("Red Channel (Saturated)")
axes[1].imshow(G, cmap='Greens'); axes[1].set_title("Green Channel (High Contrast)")
axes[2].imshow(B_ch, cmap='Blues'); axes[2].set_title("Blue Channel (Noisy)")
for ax in axes: ax.axis('off')
plt.show()"""),

    # SECTION 3
    nbf.v4.new_markdown_cell("## SECTION 3 — CONTRAST ENHANCEMENT\nEnhances structural visibility."),
    nbf.v4.new_code_cell("""# 8. Histogram Equalization (HE)
he_img = cv2.equalizeHist(gray_img)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_img, cmap='gray'); axes[0].set_title("Grayscale"); axes[0].axis('off')
axes[1].imshow(he_img, cmap='gray'); axes[1].set_title("Global Histogram Eq"); axes[1].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 9. CLAHE (Contrast Limited Adaptive Histogram Equalization)\nGlobal HE over-amplifies noise in flat regions. CLAHE operates on small tiles (e.g., 8x8) to enhance local contrast without artificial noise spikes. Crucial for retinal images to highlight vessels unevenly illuminated."),
    nbf.v4.new_code_cell("""# 9. CLAHE on grayscale
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray_img)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(he_img, cmap='gray'); axes[0].set_title("Global HE (Noisy)"); axes[0].axis('off')
axes[1].imshow(clahe_gray, cmap='gray'); axes[1].set_title("CLAHE clip=2.0 (Better details)"); axes[1].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 10. Ben Graham Preprocessing\nAdds weighted blurred image. Highly effective at normalizing variations in lighting common in fundus images."),
    nbf.v4.new_code_cell("""# 10. Ben Graham preprocessing
sigmaX = 10
bg_blurred = cv2.GaussianBlur(sample_img, (0, 0), sigmaX)
ben_graham_img = cv2.addWeighted(sample_img, 4, bg_blurred, -4, 128)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sample_img); axes[0].set_title("Original RGB"); axes[0].axis('off')
axes[1].imshow(ben_graham_img); axes[1].set_title("Ben Graham Processed"); axes[1].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 11. CLAHE on LAB L-channel\nEnhances contrast without modifying colors (Hue/Saturation)."),
    nbf.v4.new_code_cell("""# 11. CLAHE on LAB L-channel
l_clahe = clahe.apply(L)
lab_enhanced = cv2.merge((l_clahe, A, B))
rgb_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sample_img); axes[0].set_title("Original RGB"); axes[0].axis('off')
axes[1].imshow(rgb_clahe); axes[1].set_title("CLAHE on LAB L-channel"); axes[1].axis('off')
plt.show()"""),

    # SECTION 4
    nbf.v4.new_markdown_cell("## SECTION 4 — NOISE REDUCTION & FILTERING\nRetinal images often contain sensor noise which can interfere with edge detection and lesion segmentation."),
    nbf.v4.new_code_cell("""# 12. Gaussian Blur
blur_gauss = cv2.GaussianBlur(sample_img, (5, 5), 0)

# 13. Median Filter: Best for salt-and-pepper noise without blurring strong edges
blur_median = cv2.medianBlur(sample_img, 5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sample_img); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].imshow(blur_gauss); axes[1].set_title("Gaussian Blur (5x5)"); axes[1].axis('off')
axes[2].imshow(blur_median); axes[2].set_title("Median Filter (5x5)"); axes[2].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 14 & 15. Bilateral and Non-Local Means\nBilateral filter smooths while preserving edges (useful for vessel walls). NLM is excellent but computationally heavy."),
    nbf.v4.new_code_cell("""# 14. Bilateral Filter
blur_bilateral = cv2.bilateralFilter(sample_img, d=9, sigmaColor=75, sigmaSpace=75)

# 15. Non-Local Means
blur_nlm = cv2.fastNlMeansDenoisingColored(sample_img, None, 10, 10, 7, 21)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sample_img); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].imshow(blur_bilateral); axes[1].set_title("Bilateral Filter"); axes[1].axis('off')
axes[2].imshow(blur_nlm); axes[2].set_title("Non-Local Means"); axes[2].axis('off')
plt.show()"""),

    # SECTION 5
    nbf.v4.new_markdown_cell("## SECTION 5 — EDGE DETECTION & FEATURE EXTRACTION\nFinding boundaries of vessels and lesions."),
    nbf.v4.new_code_cell("""# 16. Sobel Edge Detection
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_comb = cv2.magnitude(sobelx, sobely)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.abs(sobelx), cmap='gray'); axes[0].set_title("Sobel X"); axes[0].axis('off')
axes[1].imshow(np.abs(sobely), cmap='gray'); axes[1].set_title("Sobel Y"); axes[1].axis('off')
axes[2].imshow(sobel_comb, cmap='gray'); axes[2].set_title("Sobel Combined"); axes[2].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 17. Canny Edge Detection & 18. Laplacian of Gaussian (LoG)\nCanny uses non-maximum suppression and hysteresis thresholding. LoG captures zero-crossings which relate to edges."),
    nbf.v4.new_code_cell("""# 17. Canny Edge Detection
canny_edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

# 18. Laplacian of Gaussian (LoG)
blur_for_log = cv2.GaussianBlur(gray_img, (3, 3), 0)
log_edges = cv2.Laplacian(blur_for_log, cv2.CV_64F)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(canny_edges, cmap='gray'); axes[0].set_title("Canny Edges"); axes[0].axis('off')
axes[1].imshow(np.abs(log_edges), cmap='gray'); axes[1].set_title("LoG Edges"); axes[1].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 19. Gabor Filter Bank\nGabor filters are excellent for responding to oriented features (like vessels)."),
    nbf.v4.new_code_cell("""# 19. Gabor Filter Bank
gabor_responses = []
for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]: # 0, 45, 90, 135 deg
    ksize = 31
    kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
    gabor_responses.append((filtered, theta * 180 / np.pi))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, (resp, angle) in enumerate(gabor_responses):
    axes[idx].imshow(resp, cmap='gray')
    axes[idx].set_title(f"Gabor {angle} deg")
    axes[idx].axis('off')
plt.show()"""),

    # SECTION 6
    nbf.v4.new_markdown_cell("## SECTION 6 — MORPHOLOGICAL OPERATIONS\nMorphology helps in cleaning segmented mask outputs (e.g., vessel gaps, disconnected components)."),
    nbf.v4.new_code_cell("""# Binarize a green channel based on threshold to demonstrate
_, thresh_green = cv2.threshold(G, 100, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 20. Erosion and Dilation
erosion = cv2.erode(thresh_green, kernel, iterations=1)
dilation = cv2.dilate(thresh_green, kernel, iterations=1)

# 21. Opening and Closing
opening = cv2.morphologyEx(thresh_green, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh_green, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(thresh_green, cmap='gray'); axes[0, 0].set_title("Thresholded Green"); axes[0, 0].axis('off')
axes[0, 1].imshow(erosion, cmap='gray'); axes[0, 1].set_title("Erosion (shrinks connected)"); axes[0, 1].axis('off')
axes[0, 2].imshow(dilation, cmap='gray'); axes[0, 2].set_title("Dilation (expands connected)"); axes[0, 2].axis('off')

axes[1, 0].imshow(opening, cmap='gray'); axes[1, 0].set_title("Opening (removes small noise)"); axes[1, 0].axis('off')
axes[1, 1].imshow(closing, cmap='gray'); axes[1, 1].set_title("Closing (fills small holes)"); axes[1, 1].axis('off')
axes[1, 2].axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 22-24. Top-Hat, Black-Hat, Morphological Gradient\nTop-hat enhances bright structures (exudates). Black-hat enhances dark structures (hemorrhages/vessels)."),
    nbf.v4.new_code_cell("""# 22. Top-Hat Transform (white top-hat for bright lesions)
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
tophat = cv2.morphologyEx(G, cv2.MORPH_TOPHAT, kernel_large)

# 23. Black-Hat Transform (for dark lesions)
blackhat = cv2.morphologyEx(G, cv2.MORPH_BLACKHAT, kernel_large)

# 24. Morphological Gradient
gradient = cv2.morphologyEx(G, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tophat, cmap='gray'); axes[0].set_title("Top-Hat (Exudate Extraction)"); axes[0].axis('off')
axes[1].imshow(blackhat, cmap='gray'); axes[1].set_title("Black-Hat (Hemorrhage/Vessels)"); axes[1].axis('off')
axes[2].imshow(gradient, cmap='gray'); axes[2].set_title("Gradient (Edge Enhancement)"); axes[2].axis('off')
plt.show()"""),

    # SECTION 7
    nbf.v4.new_markdown_cell("## SECTION 7 — BLOOD VESSEL SEGMENTATION\nTraditional methods like Frangi Vesselness Filter."),
    nbf.v4.new_code_cell("""# 25. Green channel + CLAHE + Otsu
g_clahe = clahe.apply(G)
_, mask_otsu = cv2.threshold(g_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 26 & 27. Frangi & Sato filters (using skimage)
bg_float = g_clahe.astype(np.float32) / 255.0
frangi_vessels = frangi(bg_float, sigmas=range(1, 6), alpha=0.5, beta=0.5, gamma=15, black_ridges=True)
sato_vessels = sato(bg_float, sigmas=range(1, 6), black_ridges=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(mask_otsu, cmap='gray'); axes[0].set_title("Otsu Thresholding")
axes[1].imshow(frangi_vessels, cmap='gray'); axes[1].set_title("Frangi Filter")
axes[2].imshow(sato_vessels, cmap='gray'); axes[2].set_title("Sato Tubeness")
for ax in axes: ax.axis('off')
plt.show()"""),
    nbf.v4.new_markdown_cell("### 28-29. Hessian enhancement & Final Segmentation Result Overlay"),
    nbf.v4.new_code_cell("""# Threshold Frangi for final mask
thresh_val = 0.05
frangi_mask = (frangi_vessels > thresh_val).astype(np.uint8) * 255
frangi_clean = cv2.morphologyEx(frangi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

# Overlay on original
vessels_overlay = sample_img.copy()
vessels_overlay[frangi_clean > 0] = [255, 0, 0] # Red overlay

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(frangi_clean, cmap='gray'); axes[0].set_title("Final Vessel Mask"); axes[0].axis('off')
axes[1].imshow(vessels_overlay); axes[1].set_title("Vessel Segmentation Overlay"); axes[1].axis('off')
plt.show()"""),

    # SECTION 8
    nbf.v4.new_markdown_cell("## SECTION 8 — LESION DETECTION (Pathology Markers)"),
    nbf.v4.new_code_cell("""# 30. Exudates (Bright lesions)
_, ex_thresh = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
ex_mask = cv2.morphologyEx(ex_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=2)

# 31. Hemorrhages (Dark lesions)
bh_blur = cv2.GaussianBlur(blackhat, (5,5), 0)
hem_mask = cv2.adaptiveThreshold(bh_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -3)
hem_mask = cv2.morphologyEx(hem_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

# 32. Microaneurysms
kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opened = cv2.morphologyEx(G, cv2.MORPH_OPEN, kernel_ma)
diff = cv2.subtract(opened, G)
_, ma_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
ma_mask = cv2.morphologyEx(ma_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

# 33. Optic Disc (Brightest large blob)
_, od_thresh = cv2.threshold(G, 220, 255, cv2.THRESH_BINARY)
od_mask = cv2.dilate(od_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

# Overlay
lesions_overlay = sample_img.copy()
lesions_overlay[ex_mask > 0] = [255, 255, 0] # Yellow Exudates
lesions_overlay[hem_mask > 0] = [255, 0, 0]  # Red Hemorrhages
lesions_overlay[ma_mask > 0] = [255, 0, 255] # Magenta MA
lesions_overlay[od_mask > 0] = [0, 255, 255] # Cyan Optic disc

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(sample_img); axes[0].set_title("Original RGB"); axes[0].axis('off')
axes[1].imshow(lesions_overlay); axes[1].set_title("Detected Lesions Overlay"); axes[1].axis('off')
plt.show()"""),

    # SECTION 9
    nbf.v4.new_markdown_cell("## SECTION 9 — ADVANCED PREPROCESSING PIPELINE\nComplete pipeline workflow."),
    nbf.v4.new_code_cell("""# 34. Complete Pipeline function
def full_pipeline_demo(img):
    img = cv2.resize(img, (512, 512))
    # Ben Graham
    bg_blurred = cv2.GaussianBlur(img, (0, 0), 10)
    bg = cv2.addWeighted(img, 4, bg_blurred, -4, 128)
    # CLAHE LAB L
    lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)
    L, A, B_ = cv2.split(lab)
    l_clahe = clahe.apply(L)
    lab_enhanced = cv2.merge((l_clahe, A, B_))
    rgb_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    # Denoise
    final_img = cv2.fastNlMeansDenoisingColored(rgb_clahe, None, 3, 3, 7, 21)
    return final_img

# 36. Apply pipeline to one image of each DR class
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    img = samples[i][0]
    processed = full_pipeline_demo(img)
    
    axes[0, i].imshow(cv2.resize(img, (512, 512)))
    axes[0, i].set_title(f"Class {i} Original", color='cyan')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(processed)
    axes[1, i].set_title("Processed", color='lime')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()"""),

    # SECTION 10
    nbf.v4.new_markdown_cell("## SECTION 10 — DATA AUGMENTATION PREVIEW\nDeep learning models require data augmentation arrays to increase robustness."),
    nbf.v4.new_code_cell("""import albumentations as A

# 37. Augmentation grid
aug_pipeline = A.Compose([
    A.RandomHorizontalFlip(p=1.0),
    A.RandomVerticalFlip(p=1.0),
    A.Rotate(limit=30, p=1.0),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=1.0),
    A.RandomCrop(width=400, height=400, p=1.0),
    A.GridDistortion(p=1.0),
    A.ElasticTransform(alpha=1, sigma=50, p=1.0)
])

base_img = cv2.resize(samples[0][0], (512, 512))
augs = [base_img]
for _ in range(7):
    augs.append(aug_pipeline(image=base_img)['image'])

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, aug_img in enumerate(augs):
    axes[i].imshow(aug_img)
    axes[i].set_title("Original" if i==0 else f"Augmentation {i}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()"""),

    # SECTION 11
    nbf.v4.new_markdown_cell("## SECTION 11 — DATASET ANALYSIS & STATISTICS\nInvestigating bias and distributions across the 5 grades."),
    nbf.v4.new_code_cell("""# 38. Class distribution
class_counts = train_df['diagnosis'].value_counts().sort_index()
percentages = (class_counts / class_counts.sum()) * 100

plt.figure(figsize=(10, 5))
bars = plt.bar(CLASS_NAMES, class_counts, color=['#22c55e', '#84cc16', '#f59e0b', '#ef4444', '#991b1b'])
plt.title("Class Distribution in APTOS 2019 Train Data")
plt.ylabel("Number of images")

for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval}\\n({percentages[i]:.1f}%)", ha='center', color='white', fontweight='bold')
plt.show()"""),
    
    nbf.v4.new_code_cell("""# 41. Pixel Intensity Violin Plots
def extract_mean_intensities(df, num_samples=200):
    subset = df.sample(num_samples, random_state=42)
    rows = []
    for _, row in subset.iterrows():
        img = cv2.imread(os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png"))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows.append({
                'Grade': CLASS_NAMES[row['diagnosis']],
                'Mean Intensity': gray.mean()
            })
    return pd.DataFrame(rows)

intensities_df = extract_mean_intensities(train_df, 300)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Grade', y='Mean Intensity', data=intensities_df, palette='viridis')
plt.title("Mean Pixel Intensity Distribution per DR Grade")
plt.show()"""),

    # Mock Patient Report
    nbf.v4.new_markdown_cell("## FINAL REPORT PREVIEW\nMockup of how these combined techniques surface as a medical dashboard."),
    nbf.v4.new_code_cell("""# Mock Patient Report
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 3)

ax_main = fig.add_subplot(gs[:, 0])
ax_main.imshow(sample_img)
ax_main.set_title("Patient Retinal Scan (Original)", color='white')
ax_main.axis('off')

ax_proc = fig.add_subplot(gs[0, 1])
ax_proc.imshow(full_pipeline_demo(sample_img))
ax_proc.set_title("AI Preprocessed", color='cyan')
ax_proc.axis('off')

ax_ves = fig.add_subplot(gs[1, 1])
ax_ves.imshow(vessels_overlay)
ax_ves.set_title("Vessel & Lesion Map", color='cyan')
ax_ves.axis('off')

ax_text = fig.add_subplot(gs[:, 2])
ax_text.axis('off')

report_text = f\"\"\"
PATIENT DR REPORT (Mock)
--------------------------
ID: #{samples[2][2].upper()[:8]}
Assessment: Moderate DR (Grade 2)

AI Findings:
- Vessel Density: Abnormal branching
- Hard Exudates: Detected
- Hemorrhages: Detected

Risk Level: Moderate
Blindness Risk: ~35%

Recommendations:
- Refer to ophthalmologist within 3-6 mos.
- Optimize glycemic control (HbA1c).
\"\"\"
ax_text.text(0.1, 0.5, report_text, color='lime', fontsize=14, family='monospace', va='center')

plt.tight_layout()
plt.show()""")
]

with open('D:/VIT/Projects/blindness-ai/blindness_ai/notebooks/preprocessing_showcase.ipynb', 'w') as f:
    nbf.write(nb, f)
