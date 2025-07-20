import cv2
import numpy as np

def get_red_shades():
    """ Returns a dictionary of red shades with their RGB values. """
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
    """ Converts a BGR color to LAB space for better perceptual distance measurement. """
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
    mask = mask1 + mask2

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

        # Find the closest red shade using LAB color distance (perceptual difference)
        closest_shade = min(
            red_shades_lab.keys(),
            key=lambda k: np.linalg.norm(np.array(avg_color_lab) - np.array(red_shades_lab[k]))
        )
        detected_colors.add(closest_shade)

    return detected_colors

def print_color_moods(detected_colors):
    """ Prints detected red shades with their associated moods. """
    color_moods = {
        "Bright Red": "Energetic, Passionate, and Exciting.",
        "Lipstick Red": "Bold, Confident, and Glamorous.",
        "Rusty Red": "Warm, Earthy, and Vintage.",
        "Fire Engine Red": "Urgent, Attention-Grabbing, and Intense.",
        "Deep Red": "Mysterious, Sophisticated, and Powerful.",
        "Cherry Red": "Playful, Sweet, and Romantic.",
        "Scarlet": "Daring, Fiery, and Ambitious.",
        "Rose Red": "Elegant, Feminine, and Warm.",
        "Crimson": "Dramatic, Luxurious, and Passionate.",
        "Maroon": "Classic, Refined, and Traditional.",
        "Vermilion": "Warm, Friendly, and Energetic.",
        "Burgundy": "Mature, Deep, and Sophisticated.",
        "Carmine": "Rich, Sensual, and Artistic.",
        "Coral Red": "Cheerful, Fresh, and Lively.",
        "Ruby Red": "Luxurious, Romantic, and Strong.",
        "Blood Red": "Intense, Strong, and Passionate.",
        "Tomato Red": "Friendly, Warm, and Approachable.",
        "Salmon Red": "Soft, Playful, and Comforting.",
        "Dark Red": "Serious, Deep, and Mysterious.",
        "Indian Red": "Exotic, Cultural, and Warm."
    }

    overall_mood = set()
    print("\nDetected Red Shades and Their Moods:")
    for color in detected_colors:
        mood = color_moods.get(color, "No specific mood found.")
        print(f"Detected {color}: {mood}")
        overall_mood.add(mood)

    print("\nOverall Mood of the Image:")
    print(" ".join(overall_mood))

if __name__ == "__main__":
    image_path = "images/TEST.jpg"  
    detected_colors = detect_reds(image_path)
    print_color_moods(detected_colors)