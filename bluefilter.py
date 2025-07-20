import cv2
import numpy as np

def enhance_blue_weak(image_path, output_path="enhanced_blue.png"):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 🌟 Shift blue tones to cyan (for better visibility)
    B = cv2.add(B, 20)  # Increase blue channel intensity slightly
    
    # 🌟 Boost contrast using histogram equalization
    L = cv2.equalizeHist(L)
    
    # Merge adjusted channels
    enhanced_lab = cv2.merge([L, A, B])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 🌟 Replace blue with purple for better visibility
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Replace blue with purple (BGR value for purple = [128, 0, 128])
    enhanced_img[mask > 0] = [128, 0, 128]

    # 🌟 Add edge highlighting for better object recognition
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
    cv2.imwrite(output_path, enhanced_img)  # Save enhanced image
    cv2.imshow("Enhanced for Blue-weak Vision", enhanced_img_resized)  # Display the resized image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "images/blueberry.jpeg"
enhance_blue_weak(image_path)
