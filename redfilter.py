import cv2
import numpy as np

def enhance_red_weak(image_path, output_path="enhanced_red.png"):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 🌟 Boost red tones (increase A channel for red shift)
    A = cv2.add(A, 30)  # Increase red intensity
    
    # 🌟 Boost contrast using histogram equalization
    L = cv2.equalizeHist(L)
    
    # Merge adjusted channels
    enhanced_lab = cv2.merge([L, A, B])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 🌟 Shift red tones to orange or pink for better visibility
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    
    # Create mask for red
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Replace red with orange (BGR for orange = [0, 165, 255])
    enhanced_img[mask > 0] = [0, 165, 255]

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
    cv2.imshow("Enhanced for Red-weak Vision", enhanced_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "images/test5.jpg"
enhance_red_weak(image_path)
