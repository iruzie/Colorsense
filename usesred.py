import cv2
import numpy as np

def get_red_shades():
    
    return {
        "Bright Red": (255, 0, 13),
        "Lipstick Red": (192, 2, 47),
        "Rusty Red": (175, 47, 13),
        "Fire Engine Red": (254, 0, 2),
        "Deep Red": (154, 2, 0),
        "Cherry Red": (247, 2, 42),
        "Scarlet": (190, 1, 25),
        "Rose Red": (190, 1, 60),
        "Crimson": (153, 0, 0),
        "Maroon": (128, 0, 0),
        "Vermilion": (227, 66, 52),
        "Burgundy": (128, 0, 32),
        "Carmine": (150, 0, 24),
        "Coral Red": (255, 64, 64),
        "Ruby Red": (155, 17, 30),
        "Blood Red": (102, 0, 0),
        "Tomato Red": (255, 99, 71),
        "Salmon Red": (250, 128, 114),
        "Dark Red": (139, 0, 0),
        "Indian Red": (205, 92, 92)
    }

def convert_to_lab(color_bgr):
    """ Converts a BGR color to LAB space for accurate perceptual color matching. """
    color_bgr = np.uint8([[color_bgr]])  # Convert to NumPy array
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)  # Convert to LAB
    return tuple(map(int, color_lab[0][0]))

def detect_reds(image_path):
    """ Detects red shades in an image and matches them to the closest predefined shade. """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the red color range in HSV (Red has two ranges in HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2  # Combine both red masks

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_shades = get_red_shades()
    red_shades_lab = {color: convert_to_lab(value) for color, value in red_shades.items()}
    
    detected_colors = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:  # Ignore small areas (noise)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]  # Extract average BGR color
        avg_color_lab = convert_to_lab(tuple(map(int, avg_color)))

        # Find the closest red shade using LAB color distance
        closest_shade = min(
            red_shades_lab.keys(),
            key=lambda k: np.linalg.norm(np.array(avg_color_lab) - np.array(red_shades_lab[k]))
        )
        detected_colors.add(closest_shade)

    return detected_colors

def print_color_uses(detected_colors):
    """ Prints detected red shades with their common uses. """
    color_uses = {
        "Bright Red": "Used in high-energy designs, branding, and warning signs.",
        "Lipstick Red": "Common in cosmetics, fashion, and bold statement pieces.",
        "Rusty Red": "Popular in vintage themes, autumn palettes, and home decor.",
        "Fire Engine Red": "Seen in emergency vehicles, alarms, and safety signs.",
        "Deep Red": "Used in luxury branding, formal wear, and classic themes.",
        "Cherry Red": "Common in food branding, vibrant fashion, and passion themes.",
        "Scarlet": "Popular in sports branding, national flags, and intense visuals.",
        "Rose Red": "Used in romance themes, floral designs, and beauty products.",
        "Crimson": "Found in university branding, regal aesthetics, and deep tones.",
        "Maroon": "Popular in academia, formal wear, and elegant interiors.",
        "Vermilion": "Used in cultural art, Asian designs, and bold branding.",
        "Burgundy": "Common in wine branding, high-end fashion, and formal decor.",
        "Carmine": "Used in makeup, traditional art, and bold clothing.",
        "Coral Red": "Popular in summer fashion, tropical themes, and fresh aesthetics.",
        "Ruby Red": "Found in jewelry, luxury branding, and passionate themes.",
        "Blood Red": "Used in gothic themes, horror aesthetics, and deep emotions.",
        "Tomato Red": "Common in food branding, fresh produce, and kitchen designs.",
        "Salmon Red": "Seen in soft pastels, warm interiors, and beachwear.",
        "Dark Red": "Popular in historical themes, vintage aesthetics, and classic branding.",
        "Indian Red": "Used in rustic themes, earthy tones, and cultural designs."
    }

    for color in detected_colors:
        print(f"Detected {color}: {color_uses.get(color, 'No specific use found.')}")

if __name__ == "__main__":
    image_path = "images/TEST.jpg"  
    detected_colors = detect_reds(image_path)
    print_color_uses(detected_colors)
