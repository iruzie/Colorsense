import cv2
import numpy as np

# Green shades dictionary with RGB values
COLOR_SHADES = {
    "Bright Green": (1, 255, 7),
    "Kelly Green": (2, 171, 46),
    "Lime Green": (137, 254, 5),
    "Forest Green": (6, 71, 12),
    "Shamrock Green": (2, 193, 77),
    "Emerald Green": (2, 143, 30),
    "Olive Green": (103, 122, 4),
    "Jungle Green": (4, 130, 67),
    "Moss Green": (138, 154, 91),
    "Neon Green": (57, 255, 20),
    "Dark Green": (0, 100, 0),
    "Sea Green": (46, 139, 87),
    "Spring Green": (0, 255, 127),
    "Mint Green": (152, 255, 152),
    "Hunter Green": (53, 94, 59),
    "Teal Green": (0, 128, 128),
    "Fern Green": (79, 121, 66),
    "Army Green": (75, 83, 32),
    "Chartreuse": (127, 255, 0),
    "Pine Green": (1, 121, 111)
}

# Convert RGB to HEX
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def closest_color_name(detected_rgb):
    min_distance = float("inf")
    closest_shade = "Unknown Shade"
    closest_hex = ""

    for shade_name, rgb in COLOR_SHADES.items():
        distance = np.linalg.norm(np.array(detected_rgb) - np.array(rgb))
        if distance < min_distance:
            min_distance = distance
            closest_shade = shade_name
            closest_hex = rgb_to_hex(rgb)

    return closest_shade, closest_hex

def display_shade_info(event, x, y, flags, param):
    contours, output_image, original_image = param
    if event == cv2.EVENT_MOUSEMOVE:
        found_contour = False
        output_image_copy = output_image.copy()

        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                b, g, r = original_image[y, x]
                detected_rgb = (r, g, b)
                shade_name, hex_code = closest_color_name(detected_rgb)

                # Draw black border around the contour
                cv2.drawContours(output_image_copy, [contour], -1, (0, 0, 0), 2)

                # Display shade name and hex code
                text_position = (x + 10, y - 10)
                hex_position = (x + 10, y + 10)

                # Black outline for better visibility
                cv2.putText(output_image_copy, f"{shade_name}", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output_image_copy, f"{hex_code}", hex_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)

                # Yellow fill text
                cv2.putText(output_image_copy, f"{shade_name}", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(output_image_copy, f"{hex_code}", hex_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

                found_contour = True
                break
        
        cv2.imshow("Detected Green Colors", output_image_copy if found_contour else output_image)

def detect_green(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
        return None, None, None

    image_resized = cv2.resize(image, (500, 500))
    original_image = image_resized.copy()

    # Convert to HSV for green detection
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Define a broader HSV range for green shades
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find edges and contours
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    min_contour_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    output_image = image_resized.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 0), 2)  # Black border

    return output_image, contours, original_image

if __name__ == "__main__":
    output_image, contours, original_image = detect_green("images/greent.jpg")
    if output_image is None:
        exit()

    cv2.imshow("Detected Green Colors", output_image)
    cv2.setMouseCallback("Detected Green Colors", display_shade_info, param=(contours, output_image, original_image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
