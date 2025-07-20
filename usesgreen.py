import cv2
import numpy as np

def get_green_shades():
    """Returns a dictionary of green shades with their RGB values."""
    return {
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

def convert_to_lab(color_bgr):
    """Converts a BGR color to LAB space for accurate perceptual color matching."""
    color_bgr = np.uint8([[color_bgr]])  # Convert to NumPy array
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)  # Convert to LAB
    return tuple(map(int, color_lab[0][0]))

def detect_greens(image_path):
    """Detects green shades in an image and matches them to the closest predefined shade."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the green color range in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get predefined green shades and convert them to LAB
    green_shades = get_green_shades()
    green_shades_lab = {color: convert_to_lab(value) for color, value in green_shades.items()}
    
    detected_colors = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:  # Ignore small areas (noise)
            continue

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        
        # Calculate the average color of the region
        avg_color = cv2.mean(roi)[:3]  # Extract average BGR color
        avg_color_lab = convert_to_lab(tuple(map(int, avg_color)))

        # Find the closest green shade using LAB color distance
        closest_shade = min(
            green_shades_lab.keys(),
            key=lambda k: np.linalg.norm(np.array(avg_color_lab) - np.array(green_shades_lab[k]))
        )
        detected_colors.add(closest_shade)

    return detected_colors

def print_color_uses(detected_colors):
    """Prints detected green shades with their common uses."""
    color_uses = {
        "Bright Green": "Used in fresh, energetic designs, nature themes, and tech branding.",
        "Kelly Green": "Common in sports team uniforms, St. Patrick’s Day themes, and landscapes.",
        "Lime Green": "Popular in advertising, youth brands, and eco-friendly marketing.",
        "Forest Green": "Used in military gear, nature photography, and outdoor branding.",
        "Shamrock Green": "Seen in Irish heritage, finance logos, and vibrant landscapes.",
        "Emerald Green": "Common in luxury branding, jewelry, and high-end fashion.",
        "Olive Green": "Used in military camouflage, autumn fashion, and rustic themes.",
        "Jungle Green": "Found in tropical designs, environmental campaigns, and jungle-themed visuals.",
        "Moss Green": "Popular in earthy aesthetics, nature decor, and rustic branding.",
        "Neon Green": "Used in high-visibility signs, sports branding, and neon aesthetics.",
        "Dark Green": "Common in classic suits, banking, and academic settings.",
        "Sea Green": "Ideal for coastal themes, aquariums, and relaxation spaces.",
        "Spring Green": "Seen in spring fashion, floral designs, and organic branding.",
        "Mint Green": "Used in soft pastels, cosmetics, and fresh-themed designs.",
        "Hunter Green": "Popular in hunting gear, formal menswear, and preppy fashion.",
        "Teal Green": "Found in modern web designs, medical branding, and ocean aesthetics.",
        "Fern Green": "Used in botanical themes, home decor, and organic branding.",
        "Army Green": "Common in military clothing, survival gear, and rugged fashion.",
        "Chartreuse": "Popular in bold fashion, bright advertising, and modern designs.",
        "Pine Green": "Seen in Christmas decorations, nature themes, and eco-conscious branding."
    }

    for color in detected_colors:
        print(f"Detected {color}: {color_uses.get(color, 'No specific use found.')}")

if __name__ == "__main__":
    # Path to the input image
    image_path = "images/apple.jpg"  # Replace with your image path
    
    # Detect green shades in the image
    detected_colors = detect_greens(image_path)
    
    # Print the uses of the detected green shades
    print_color_uses(detected_colors)