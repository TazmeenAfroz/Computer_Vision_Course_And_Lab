# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
img = cv2.imread('star.jpeg', cv2.IMREAD_GRAYSCALE)

image_h, image_w = img.shape

sigma_base = 0.707
k = np.sqrt(2)
octaves = []

for octave_i in range(4):
    
    if octave_i == 0:
        current_img = img
    else:
       
        current_img = cv2.resize(current_img, (current_img.shape[1]//2, current_img.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    
    sigma_values = []
    gaussian_images = []
    
    
    for scale_i in range(5):
        current_sigma = sigma_base * (k ** scale_i)
        sigma_values.append(current_sigma)
        
        blurred_img = cv2.GaussianBlur(current_img, (0, 0), sigmaX=current_sigma, sigmaY=current_sigma, borderType=cv2.BORDER_DEFAULT)
        gaussian_images.append(blurred_img)
    
    
    octaves.append({
        'gaussian_images': gaussian_images,
        'sigma_values': sigma_values,
        'shape': current_img.shape
    })

# %%



fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('SIFT Gaussian Pyramid - 4 Octaves × 5 Scales', fontsize=16)

for octave_i in range(4):
    for scale_i in range(5):
        ax = axes[octave_i, scale_i]
        img_to_show = octaves[octave_i]['gaussian_images'][scale_i]
        sigma_val = octaves[octave_i]['sigma_values'][scale_i]
        
        ax.imshow(img_to_show, cmap='gray')
        ax.set_title(f'Octave {octave_i}, σ={sigma_val:.3f}')
        ax.axis('off')

plt.tight_layout()
plt.show()

# %%

dog_octaves = []

for octave_i in range(4):
    dog_images = []
    
    
    for scale_i in range(4):  
        img1 = octaves[octave_i]['gaussian_images'][scale_i]
        img2 = octaves[octave_i]['gaussian_images'][scale_i + 1]
        
       
        dog = img2 - img1 
        dog_images.append(dog)
    
    dog_octaves.append({
        'dog_images': dog_images,
        'shape': octaves[octave_i]['shape']
    })


# %%

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Difference of Gaussians (DoG) - 4 Octaves × 4 DoG Images', fontsize=16)

for octave_i in range(4):
    for dog_i in range(4):
        ax = axes[octave_i, dog_i]
        dog_img = dog_octaves[octave_i]['dog_images'][dog_i]
        
       
        ax.imshow(dog_img, cmap='gray')
        ax.set_title(f'Octave {octave_i}, DoG {dog_i}')
        ax.axis('off')

plt.tight_layout()
plt.show()


# %%

for octave_i, octave_data in enumerate(dog_octaves):
    print(f"\nOctave {octave_i}:")
    print(f"  Shape: {octave_data['shape']}")
    

# %%
edge_threshold = 0.1
contrast_threshold = 0.04
keypoints = []  # list of tuples (x, y, sigma, octave, dog)

for octave_i in range(4):
    dog_images = dog_octaves[octave_i]['dog_images']
    sigma_values = octaves[octave_i]['sigma_values']
    h, w = dog_octaves[octave_i]['shape']

    for dog_i in range(1, 3):  # Skip the first and last DoG
        prev_dog = dog_images[dog_i - 1]
        current_dog = dog_images[dog_i]
        next_dog = dog_images[dog_i + 1]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center_value = current_dog[y, x]

                cube = np.stack([
                    prev_dog[y-1:y+2, x-1:x+2],
                    current_dog[y-1:y+2, x-1:x+2],
                    next_dog[y-1:y+2, x-1:x+2]
                ])

                if (center_value == cube.max() or center_value == cube.min()) and abs(center_value) > contrast_threshold:
                    # Hessian edge check
                    Dxx = current_dog[y, x+1] + current_dog[y, x-1] - 2 * center_value
                    Dyy = current_dog[y+1, x] + current_dog[y-1, x] - 2 * center_value
                    Dxy = (current_dog[y+1, x+1] - current_dog[y+1, x-1]
                           - current_dog[y-1, x+1] + current_dog[y-1, x-1]) / 4.0

                    detH = Dxx * Dyy - Dxy**2
                    traceH = Dxx + Dyy

                    if detH > 0:
                        r = (traceH**2) / detH
                        r_thresh = ((edge_threshold + 1)**2) / edge_threshold
                        if r < r_thresh:
                            # Get corresponding sigma value for this DoG
                            sigma = sigma_values[dog_i]
                            x_real = x * (2 ** octave_i)
                            y_real = y * (2 ** octave_i)
                            keypoints.append((x_real, y_real, sigma, octave_i, dog_i))

    print(f"Octave {octave_i}: {len(keypoints)} keypoints so far")

print(f"\nTotal keypoints detected: {len(keypoints)}")


print("\nSample of keypoints with σ values (x, y, σ, octave, dog):")
for i, (x, y, sigma, octave, dog) in enumerate(keypoints[:20]):  # print first 20
    print(f"Keypoint {i+1}: x={x:.2f}, y={y:.2f}, σ={sigma:.3f}, octave={octave}, dog={dog}")



# %%
plt.figure(figsize=(8, 8))
image_rgb = cv2.cvtColor(cv2.imread('star.jpeg'), cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Detected Keypoints with Scale")
plt.axis('off')

for (x, y, sigma, _, _) in keypoints:
    plt.scatter(x, y, s=(sigma * 20), c='lime', alpha=0.5, edgecolors='black')

plt.show()


# %%
def compute_gradient_magnitude_orientation(image):
    """Compute gradient magnitude and orientation for the image."""
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = (np.degrees(np.arctan2(Gy, Gx)) + 360) % 360
    return magnitude, orientation


# --- STEP 2: Orientation Assignment ---
def assign_orientation_to_keypoints(octaves, keypoints, num_bins=36):
    """Assign dominant orientations to keypoints for rotation invariance."""
    oriented_keypoints = []
    bin_width = 360 / num_bins

    for (x, y, sigma, octave_i, dog_i) in keypoints:
        gaussian_image = octaves[octave_i]['gaussian_images'][dog_i]
        magnitude, orientation = compute_gradient_magnitude_orientation(gaussian_image)

        h, w = gaussian_image.shape
        radius = int(3 * sigma)

        if x < radius or y < radius or x >= (w - radius) or y >= (h - radius):
            continue

        patch_mag = magnitude[int(y - radius):int(y + radius + 1),
                              int(x - radius):int(x + radius + 1)]
        patch_ori = orientation[int(y - radius):int(y + radius + 1),
                                int(x - radius):int(x + radius + 1)]

        y_grid, x_grid = np.mgrid[-radius:radius+1, -radius:radius+1]
        gaussian_weight = np.exp(-(x_grid**2 + y_grid**2) / (2 * (1.5 * sigma)**2))

        hist = np.zeros(num_bins)
        for i in range(patch_ori.shape[0]):
            for j in range(patch_ori.shape[1]):
                angle = patch_ori[i, j]
                bin_idx = int(angle // bin_width) % num_bins
                hist[bin_idx] += patch_mag[i, j] * gaussian_weight[i, j]

        hist = np.convolve(hist, [1/3, 1/3, 1/3], mode='same')

        max_val = np.max(hist)
        dominant_bins = np.where(hist >= 0.8 * max_val)[0]

        for b in dominant_bins:
            theta = b * bin_width
            oriented_keypoints.append((x, y, sigma, octave_i, dog_i, theta))

    return oriented_keypoints


# --- STEP 3: Descriptor Formation (128D) ---
def compute_sift_descriptors(octaves, oriented_keypoints):
    """Generate 128-dimensional SIFT descriptors."""
    descriptors = []
    for (x, y, sigma, octave_i, dog_i, theta) in oriented_keypoints:
        gaussian_image = octaves[octave_i]['gaussian_images'][dog_i]
        magnitude, orientation = compute_gradient_magnitude_orientation(gaussian_image)

        h, w = gaussian_image.shape
        patch_radius = int(8 * sigma)
        if x < patch_radius or y < patch_radius or x >= (w - patch_radius) or y >= (h - patch_radius):
            continue

        patch_mag = magnitude[int(y - patch_radius):int(y + patch_radius),
                              int(x - patch_radius):int(x + patch_radius)]
        patch_ori = orientation[int(y - patch_radius):int(y + patch_radius),
                                int(x - patch_radius):int(x + patch_radius)]

        cos_t, sin_t = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        descriptor = []

        # Divide patch into 4x4 subregions
        sub_size = patch_mag.shape[0] // 4
        for i in range(4):
            for j in range(4):
                y_start = i * sub_size
                y_end = y_start + sub_size
                x_start = j * sub_size
                x_end = x_start + sub_size

                sub_mag = patch_mag[y_start:y_end, x_start:x_end]
                sub_ori = patch_ori[y_start:y_end, x_start:x_end]

                # Rotate orientations relative to keypoint orientation
                rel_ori = (sub_ori - theta + 360) % 360

                hist, _ = np.histogram(rel_ori, bins=8, range=(0, 360), weights=sub_mag)
                descriptor.extend(hist)

        descriptor = np.array(descriptor, dtype=np.float32)
        descriptor /= (np.linalg.norm(descriptor) + 1e-10)
        descriptor = np.clip(descriptor, 0, 0.2)
        descriptor /= (np.linalg.norm(descriptor) + 1e-10)

        descriptors.append(descriptor)

    return np.array(descriptors)


# --- STEP 4: Visualization ---
def visualize_oriented_keypoints(image_path, oriented_keypoints):
    img_color = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title("Keypoints with Orientation Arrows")
    plt.axis('off')

    for (x, y, sigma, octave, dog, theta) in oriented_keypoints[:200]:
        dx = 5 * np.cos(np.deg2rad(theta))
        dy = 5 * np.sin(np.deg2rad(theta))
        plt.arrow(x, y, dx, -dy, color='red', width=0.4, head_width=2)
    plt.show()


# --- STEP 5: Validation with OpenCV’s SIFT ---
def compare_with_opencv(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img_gray, None)

    print(f"\nOpenCV SIFT: {len(kps)} keypoints, Descriptor shape: {desc.shape}")
    img_sift = cv2.drawKeypoints(img_gray, kps, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_sift, cmap='gray')
    plt.title("OpenCV SIFT Keypoints")
    plt.axis('off')
    plt.show()

oriented_keypoints = assign_orientation_to_keypoints(octaves, keypoints)
descriptors = compute_sift_descriptors(octaves, oriented_keypoints)
print(f"\nTotal oriented keypoints: {len(oriented_keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")
visualize_oriented_keypoints('star.jpeg', oriented_keypoints)
compare_with_opencv('star.jpeg')





# %%



