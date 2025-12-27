import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

# === Import your implementations ===
from canny import canny     # your custom Canny function
from hough import hough_transform, draw_lines  # your custom Hough

# ---------- Fixed line drawing for skimage results ----------
def draw_skimage_lines(img, angles, dists):
    """
    Draw lines from skimage hough_line_peaks results safely
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape
    
    for angle, dist in zip(angles, dists):
        # Convert angle and distance to line parameters
        a = np.cos(angle)
        b = np.sin(angle)
        
        # Handle different cases to avoid division by zero
        if np.abs(b) < 1e-6:  # Nearly horizontal line (sin ≈ 0)
            # Line is nearly vertical: x = dist/cos(angle)
            x = int(dist / a) if np.abs(a) > 1e-6 else 0
            x = max(0, min(width-1, x))  # Clamp to image bounds
            cv2.line(img_color, (x, 0), (x, height-1), (0, 255, 0), 2)
            
        elif np.abs(a) < 1e-6:  # Nearly vertical line (cos ≈ 0)
            # Line is nearly horizontal: y = dist/sin(angle)
            y = int(dist / b) if np.abs(b) > 1e-6 else 0
            y = max(0, min(height-1, y))  # Clamp to image bounds
            cv2.line(img_color, (0, y), (width-1, y), (0, 255, 0), 2)
            
        else:  # General case
            # Calculate two points on the line
            x0 = a * dist
            y0 = b * dist
            
            # Extend line in both directions
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_color

# ---------- Parameter comparison function ----------
def compare_canny_parameters(img):
    """Compare Canny edge detection with different parameters"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Different parameter sets for comparison
    param_sets = [
        {"sigma": 0.5, "low": 30, "high": 80, "title": "σ=0.5, low=30, high=80"},
        {"sigma": 1.0, "low": 50, "high": 100, "title": "σ=1.0, low=50, high=100"},
        {"sigma": 1.5, "low": 70, "high": 140, "title": "σ=1.5, low=70, high=140"},
        {"sigma": 2.0, "low": 40, "high": 120, "title": "σ=2.0, low=40, high=120"}
    ]
    
    plt.figure(figsize=(20, 12))
    
    # Show original image
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Show custom Canny results
    for i, params in enumerate(param_sets):
        edges_custom = canny(img, sigma=params["sigma"], low=params["low"], high=params["high"])
        plt.subplot(3, 4, i + 2)
        plt.imshow(edges_custom, cmap="gray")
        plt.title(f"Custom Canny\n{params['title']}")
        plt.axis("off")
    
    # Show OpenCV Canny results for comparison
    for i, params in enumerate(param_sets):
        edges_cv = cv2.Canny(gray, params["low"], params["high"])
        plt.subplot(3, 4, i + 6)
        plt.imshow(edges_cv, cmap="gray")
        plt.title(f"OpenCV Canny\n{params['title']}")
        plt.axis("off")
    
    # Show difference (Custom - OpenCV) for first parameter set
    edges_custom_ref = canny(img, sigma=param_sets[0]["sigma"], low=param_sets[0]["low"], high=param_sets[0]["high"])
    edges_cv_ref = cv2.Canny(gray, param_sets[0]["low"], param_sets[0]["high"])
    
    plt.subplot(3, 4, 10)
    diff = np.abs(edges_custom_ref.astype(float) - edges_cv_ref.astype(float))
    plt.imshow(diff, cmap="hot")
    plt.title("Difference\n(Custom - OpenCV)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== CANNY EDGE DETECTION COMPARISON ===")
    for i, params in enumerate(param_sets):
        edges_custom = canny(img, sigma=params["sigma"], low=params["low"], high=params["high"])
        edges_cv = cv2.Canny(gray, params["low"], params["high"])
        custom_pixels = np.sum(edges_custom > 0)
        cv_pixels = np.sum(edges_cv > 0)
        print(f"Params {params['title']}:")
        print(f"  Custom: {custom_pixels} edge pixels")
        print(f"  OpenCV: {cv_pixels} edge pixels")
        print(f"  Difference: {custom_pixels - cv_pixels} pixels")

def compare_hough_parameters(img):
    """Compare Hough Transform with different parameters"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use consistent Canny parameters for fair comparison
    edges_custom = canny(img, sigma=1.0, low=50, high=100)
    edges_cv = cv2.Canny(gray, 50, 100)
    
    # Different threshold values for Hough transform
    hough_thresholds = [60, 100, 150, 200]
    
    plt.figure(figsize=(20, 10))
    
    # Show edge images
    plt.subplot(2, 5, 1)
    plt.imshow(edges_custom, cmap="gray")
    plt.title("Custom Canny Edges")
    plt.axis("off")
    
    plt.subplot(2, 5, 6)
    plt.imshow(edges_cv, cmap="gray")
    plt.title("OpenCV Canny Edges")
    plt.axis("off")
    
    print("\n=== HOUGH TRANSFORM COMPARISON ===")
    
    # Show custom Hough results with different thresholds
    for i, thresh in enumerate(hough_thresholds):
        # Custom implementation
        acc, rhos, thetas, lines = hough_transform(edges_custom, threshold=thresh)
        img_hough_custom = draw_lines(gray, lines)
        
        plt.subplot(2, 5, i + 2)
        plt.imshow(cv2.cvtColor(img_hough_custom, cv2.COLOR_BGR2RGB))
        plt.title(f"Custom Hough\nThreshold={thresh}")
        plt.axis("off")
        
        # Skimage implementation
        h, theta_vals, d = hough_line(edges_cv)
        # Adjust threshold as percentage of max accumulator value
        thresh_ratio = thresh / 200.0  # normalize to 0-1 range
        accum, angles, dists = hough_line_peaks(h, theta_vals, d, threshold=thresh_ratio * np.max(h))
        img_hough_sk = draw_skimage_lines(gray, angles, dists)
        
        plt.subplot(2, 5, i + 7)
        plt.imshow(cv2.cvtColor(img_hough_sk, cv2.COLOR_BGR2RGB))
        plt.title(f"Skimage Hough\nThreshold={thresh}")
        plt.axis("off")
        
        print(f"Threshold {thresh}:")
        print(f"  Custom: {len(lines)} lines detected")
        print(f"  Skimage: {len(angles)} lines detected")
    
    plt.tight_layout()
    plt.show()

def comprehensive_comparison(img):
    """Show a comprehensive comparison with the best parameters"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Best parameters (you can adjust these)
    best_sigma = 1.0
    best_low = 50
    best_high = 100
    best_hough_thresh = 120
    
    # Generate results
    edges_custom = canny(img, sigma=best_sigma, low=best_low, high=best_high)
    edges_cv = cv2.Canny(gray, best_low, best_high)
    
    acc, rhos, thetas, lines = hough_transform(edges_custom, threshold=best_hough_thresh)
    img_hough_custom = draw_lines(gray, lines)
    
    h, theta_vals, d = hough_line(edges_cv)
    accum, angles, dists = hough_line_peaks(h, theta_vals, d, threshold=0.6 * np.max(h))
    img_hough_sk = draw_skimage_lines(gray, angles, dists)
    
    # Display results
    plt.figure(figsize=(20, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(2, 4, 2)
    plt.imshow(edges_custom, cmap="gray")
    plt.title(f"Custom Canny\nσ={best_sigma}, {best_low}-{best_high}")
    plt.axis("off")
    
    plt.subplot(2, 4, 3)
    plt.imshow(edges_cv, cmap="gray")
    plt.title(f"OpenCV Canny\n{best_low}-{best_high}")
    plt.axis("off")
    
    plt.subplot(2, 4, 4)
    diff = np.abs(edges_custom.astype(float) - edges_cv.astype(float))
    plt.imshow(diff, cmap="hot")
    plt.title("Edge Difference\n(Custom - OpenCV)")
    plt.axis("off")
    
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(img_hough_custom, cv2.COLOR_BGR2RGB))
    plt.title(f"Custom Hough\nThreshold={best_hough_thresh}")
    plt.axis("off")
    
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(img_hough_sk, cv2.COLOR_BGR2RGB))
    plt.title(f"Skimage Hough\nAdaptive threshold")
    plt.axis("off")
    
    plt.subplot(2, 4, 8)
    plt.imshow(acc, cmap="hot")
    plt.title("Custom Hough\nAccumulator")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== COMPREHENSIVE RESULTS ===")
    print(f"Custom Canny: {np.sum(edges_custom > 0)} edge pixels")
    print(f"OpenCV Canny: {np.sum(edges_cv > 0)} edge pixels")
    print(f"Custom Hough: {len(lines)} lines detected")
    print(f"Skimage Hough: {len(angles)} lines detected")

# ---------- Main ----------
if __name__ == "__main__":
    
    # Add your image paths here
    image_paths = [
        "building.jpg",    # façade with strong lines
        "object.jpg",      # simple object
        "trees.jpg",       # natural scene
        # Add more paths as needed
    ]
    
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load {img_path}, skipping...")
                continue
                
            print(f"\n{'='*60}")
            print(f"ANALYZING: {img_path}")
            print(f"Image shape: {img.shape}")
            print(f"{'='*60}")
            
            # Resize image if too large (for better visualization)
            height, width = img.shape[:2]
            if height > 800 or width > 800:
                scale = min(800/height, 800/width)
                new_height, new_width = int(height * scale), int(width * scale)
                img = cv2.resize(img, (new_width, new_height))
                print(f"Resized to: {img.shape}")
            
            # Run different comparisons
            print("\n1. Comparing Canny parameters...")
            compare_canny_parameters(img)
            
            print("\n2. Comparing Hough parameters...")
            compare_hough_parameters(img)
            
            print("\n3. Comprehensive comparison...")
            comprehensive_comparison(img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not image_paths or all(cv2.imread(path) is None for path in image_paths):
        print("\nNo valid images found. Please check your image paths:")
        for path in image_paths:
            print(f"  - {path}")
        print("\nMake sure the image files exist in the current directory or provide full paths.")
