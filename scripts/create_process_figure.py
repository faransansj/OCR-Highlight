#!/usr/bin/env python3
"""
Create a figure showing the OCR and highlight extraction process
Uses consistent dataset with full text context visible
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def create_process_figure():
    """Create a comprehensive figure showing the extraction process"""

    # Use same base sentence with different highlight colors for consistency
    img1_path = "data/test/test_0039_orig.png"  # Yellow highlight
    img2_path = "data/test/test_0147_orig.png"  # Green highlight

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return

    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Create HSV masks to show highlight detection
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Yellow highlight detection (for img1)
    yellow_lower = np.array([15, 40, 40])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask1 = cv2.inRange(img1_hsv, yellow_lower, yellow_upper)

    # Green highlight detection (for img2)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask2 = cv2.inRange(img2_hsv, green_lower, green_upper)

    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(18, 8))

    # Row 1: Yellow highlight example
    # Original image - show full context
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img1_rgb)
    ax1.set_title('(a) Original Image\n(Yellow Highlight)', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # HSV mask
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(yellow_mask1, cmap='gray')
    ax2.set_title('(b) Color Detection\n(HSV Mask)', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # Extracted text visualization
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(img1_rgb)
    # Draw bounding box around detected area
    contours, _ = cv2.findContours(yellow_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect = mpatches.Rectangle((x, y), w, h, linewidth=4, edgecolor='yellow', facecolor='none')
        ax3.add_patch(rect)
        # Add text annotation
        ax3.text(x + w/2, y-15, 'Detected: "데이터"', fontsize=11, color='black',
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    ax3.set_title('(c) OCR Extraction\n(Tesseract)', fontsize=13, fontweight='bold')
    ax3.axis('off')

    # Row 2: Green highlight example
    # Original image - show full context
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(img2_rgb)
    ax4.set_title('(d) Original Image\n(Green Highlight)', fontsize=13, fontweight='bold')
    ax4.axis('off')

    # HSV mask
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(green_mask2, cmap='gray')
    ax5.set_title('(e) Color Detection\n(HSV Mask)', fontsize=13, fontweight='bold')
    ax5.axis('off')

    # Extracted text visualization
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(img2_rgb)
    # Draw bounding box around detected area
    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect = mpatches.Rectangle((x, y), w, h, linewidth=4, edgecolor='lime', facecolor='none')
        ax6.add_patch(rect)
        # Add text annotation
        ax6.text(x + w/2, y-15, 'Detected: "이미지 전처리"', fontsize=11, color='black',
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lime', alpha=0.8))
    ax6.set_title('(f) OCR Extraction\n(Tesseract)', fontsize=13, fontweight='bold')
    ax6.axis('off')

    # Add main title
    fig.suptitle('Highlight Text Extraction Pipeline\nOriginal → Detection → OCR',
                 fontsize=17, fontweight='bold', y=0.98)

    # Add process flow arrows
    fig.text(0.30, 0.72, '→', fontsize=45, ha='center', va='center', color='gray', weight='bold')
    fig.text(0.63, 0.72, '→', fontsize=45, ha='center', va='center', color='gray', weight='bold')
    fig.text(0.30, 0.23, '→', fontsize=45, ha='center', va='center', color='gray', weight='bold')
    fig.text(0.63, 0.23, '→', fontsize=45, ha='center', va='center', color='gray', weight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = "outputs/extraction_process_figure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved figure to: {output_path}")

    # Also save as PDF for publication
    pdf_path = "outputs/extraction_process_figure.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.close()

if __name__ == "__main__":
    create_process_figure()
