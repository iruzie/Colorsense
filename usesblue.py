import cv2
import numpy as np

def get_blue_shades():
    """ Returns a dictionary of blue shades with their RGB values. """
    return {
        "Bright Blue": (1, 101, 252),
        "Royal Blue": (5, 4, 170),
        "Deep Sky Blue": (13, 117, 248),
        "Cobalt Blue": (3, 10, 167),
        "Navy Blue": (0, 17, 70),
        "Azure": (6, 154, 243),
        "Steel Blue": (90, 125, 154),
        "Electric Blue": (6, 82, 255),
        "Turquoise": (64, 224, 208),
        "Cyan": (0, 255, 255),
        "Midnight Blue": (25, 25, 112),
        "Periwinkle": (204, 204, 255),
        "Baby Blue": (137, 207, 240),
        "Ice Blue": (214, 245, 255),
        "Dodger Blue": (30, 144, 255),
        "Prussian Blue": (0, 49, 83),
        "Sapphire Blue": (15, 82, 186),
        "Cornflower Blue": (100, 149, 237),
        "Denim Blue": (21, 96, 189),
        "Sky Blue": (135, 206, 235)
    }

def convert_to_lab(color_bgr):
    """ Converts a BGR color to LAB space for better perceptual distance measurement. """
    color_bgr = np.uint8([[color_bgr]])  # Convert to NumPy array
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)  # Convert to LAB
    return tuple(map(int, color_lab[0][0]))

def detect_blues(image_path):
    """ Detects blue shades in an image and matches them to the closest predefined shade. """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the blue color range in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_shades = get_blue_shades()
    blue_shades_lab = {color: convert_to_lab(value) for color, value in blue_shades.items()}
    
    detected_colors = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:  # Ignore small areas (noise)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]  # Extract average BGR color
        avg_color_lab = convert_to_lab(tuple(map(int, avg_color)))

        # Find the closest blue shade using LAB color distance (perceptual difference)
        closest_shade = min(
            blue_shades_lab.keys(),
            key=lambda k: np.linalg.norm(np.array(avg_color_lab) - np.array(blue_shades_lab[k]))
        )
        detected_colors.add(closest_shade)

    return detected_colors

def print_color_uses(detected_colors):
    """ Prints detected blue shades with their common uses. """
    color_uses = {
        "Bright Blue": "Used in vibrant designs, tech, and modern branding.",
        "Royal Blue": "Common in sports team logos, business suits, and formal attire.",
        "Deep Sky Blue": "Ideal for sky-related visuals, aviation, and technology.",
        "Cobalt Blue": "Popular in ceramics, glass art, and fashion.",
        "Navy Blue": "Used in military, corporate branding, and nautical themes.",
        "Azure": "Great for digital branding, ocean themes, and modern designs.",
        "Steel Blue": "Used in industrial, automotive, and architectural designs.",
        "Electric Blue": "Common in neon signs, high-energy designs, and tech products.",
        "Turquoise": "Found in jewelry, tropical designs, and wellness branding.",
        "Cyan": "Used in printing, UI design, and water-related themes.",
        "Midnight Blue": "Popular for elegant fashion, deep space visuals, and classic themes.",
        "Periwinkle": "Seen in pastels, soft aesthetics, and calming designs.",
        "Baby Blue": "Used in baby products, soft-themed branding, and casual wear.",
        "Ice Blue": "Ideal for winter-themed designs, ice-related visuals, and cool aesthetics.",
        "Dodger Blue": "Common in digital media, sports brands, and signage.",
        "Prussian Blue": "Historically used in art, dyes, and technical drawings.",
        "Sapphire Blue": "Popular in gemstones, luxury branding, and high-end fashion.",
        "Cornflower Blue": "Great for floral designs, subtle branding, and soft color schemes.",
        "Denim Blue": "Common in clothing, casual wear, and rugged themes.",
        "Sky Blue": "Used in relaxation spaces, branding, and weather-related themes."
    }

    for color in detected_colors:
        print(f"Detected {color}: {color_uses.get(color, 'No specific use found.')}")

if __name__ == "__main__":
    image_path = "images/TEST.jpg"  
    detected_colors = detect_blues(image_path)
    print_color_uses(detected_colors)