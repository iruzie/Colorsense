import cv2
import numpy as np

COLOR_SHADES = {
    "blue": {
        "Bright Blue": "#0165fc", "Royal Blue": "#0504aa", "Deep Sky Blue": "#0d75f8", "Cobalt Blue": "#030aa7",
        "Navy Blue": "#001146", "Azure": "#069af3", "Steel Blue": "#5a7d9a", "Electric Blue": "#0652ff",
        "Turquoise": "#40e0d0", "Cyan": "#00ffff", "Midnight Blue": "#191970", "Periwinkle": "#ccccff",
        "Baby Blue": "#89cff0", "Ice Blue": "#d6f5ff", "Dodger Blue": "#1e90ff", "Prussian Blue": "#003153",
        "Sapphire Blue": "#0f52ba", "Cornflower Blue": "#6495ed", "Denim Blue": "#1560bd", "Sky Blue": "#87ceeb"
    }
}

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def closest_color_name(detected_hex):
    min_distance = float("inf")
    closest_shade = "Unknown Blue Shade"
    detected_rgb = hex_to_rgb(detected_hex)
    
    for shade_name, hex_code in COLOR_SHADES["blue"].items():
        shade_rgb = hex_to_rgb(hex_code)
        distance = np.linalg.norm(np.array(detected_rgb) - np.array(shade_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_shade = shade_name
    return closest_shade, detected_hex

def display_shade_info(event, x, y, flags, param):
    contours, output_image, original_image = param
    if event == cv2.EVENT_MOUSEMOVE:
        found_contour = False
        output_image_copy = output_image.copy()
        
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                b, g, r = original_image[y, x]
                detected_hex = f"#{r:02x}{g:02x}{b:02x}"
                shade_name, hex_code = closest_color_name(detected_hex)
                
                cv2.drawContours(output_image_copy, [contour], -1, (255, 0, 255), 3)
                text_position = (x + 10, y - 10)
                hex_position = (x + 10, y + 10)
                
                cv2.putText(output_image_copy, f"{shade_name}", text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output_image_copy, f"{hex_code}", hex_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output_image_copy, f"{shade_name}", text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(output_image_copy, f"{hex_code}", hex_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                found_contour = True
                break
                
        cv2.imshow("Detected Blue Colors", output_image_copy if found_contour else output_image)

def detect_blue(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (500, 500))
    original_image = image_resized.copy()
    
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    image_denoised = cv2.medianBlur(image_blurred, 5)
    
    hsv_image = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(mask, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = image_resized.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 0), 7)
    
    return output_image, contours, original_image
if __name__ == "__main__":
    output_image, contours, original_image = detect_blue("images/test8.jpg")
    if output_image is None:
        exit()

    # Display the image and set up mouse callback to display color info
    cv2.imshow("Detected Blue Colors", output_image)
    cv2.setMouseCallback("Detected Blue Colors", display_shade_info, param=(contours, output_image, original_image))

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

