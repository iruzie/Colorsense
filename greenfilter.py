import cv2
import numpy as np

def enhance_green_weak(image_path, output_path="enhanced_green.png"):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 🌟 Boost green tones (reduce A channel for green shift)
    A = cv2.subtract(A, 30)  # Increase green intensity
    
    # 🌟 Boost contrast using histogram equalization
    L = cv2.equalizeHist(L)
    
    # Merge adjusted channels
    enhanced_lab = cv2.merge([L, A, B])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 🌟 Shift green tones to cyan or yellow for better visibility
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Replace green with cyan (BGR for cyan = [255, 255, 0])
    enhanced_img[mask > 0] = [255, 255, 0]

    # 🌟 Add edge highlighting for better object separation
    edges = cv2.Canny(enhanced_img, 100, 200)
    enhanced_img[edges > 0] = [0, 0, 0]  # Black edges

    # Resize the image for display
    height, width = enhanced_img.shape[:2]
    max_height = 500
    if height > max_height:
        scaling_factor = max_height / height
        enhanced_img_resized = cv2.resize(enhanced_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    else:
        enhanced_img_resized = enhanced_img

    # Save and display result
    cv2.imwrite(output_path, enhanced_img)
    cv2.imshow("Enhanced for Green-weak Vision", enhanced_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "images/fruits.jpg"
enhance_green_weak(image_path)
