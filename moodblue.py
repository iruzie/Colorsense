import cv2
import numpy as np
from collections import Counter

def get_blue_shades():
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

def get_color_moods():
    return {
        "Bright Blue": "Energetic and refreshing",
        "Royal Blue": "Elegant and trustworthy",
        "Deep Sky Blue": "Inspiring and uplifting",
        "Cobalt Blue": "Strong and bold",
        "Navy Blue": "Serious and authoritative",
        "Azure": "Calm and peaceful",
        "Steel Blue": "Cool and industrial",
        "Electric Blue": "Exciting and futuristic",
        "Turquoise": "Healing and tropical",
        "Cyan": "Refreshing and dynamic",
        "Midnight Blue": "Mysterious and deep",
        "Periwinkle": "Soft and dreamy",
        "Baby Blue": "Gentle and innocent",
        "Ice Blue": "Chill and serene",
        "Dodger Blue": "Modern and vibrant",
        "Prussian Blue": "Historical and intellectual",
        "Sapphire Blue": "Luxurious and confident",
        "Cornflower Blue": "Friendly and warm",
        "Denim Blue": "Casual and comfortable",
        "Sky Blue": "Relaxing and open"
    }

def convert_to_lab(color_bgr):
    color_bgr = np.uint8([[color_bgr]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    return tuple(map(int, color_lab[0][0]))

def detect_blues(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_shades = get_blue_shades()
    blue_shades_lab = {color: convert_to_lab(value) for color, value in blue_shades.items()}
    detected_colors = set()
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]
        avg_color_lab = convert_to_lab(tuple(map(int, avg_color)))
        closest_shade = min(blue_shades_lab.keys(), key=lambda k: np.linalg.norm(np.array(avg_color_lab) - np.array(blue_shades_lab[k])))
        detected_colors.add(closest_shade)
    
    return detected_colors

def determine_overall_mood(detected_colors):
    color_moods = get_color_moods()
    mood_counts = Counter(color_moods[color] for color in detected_colors if color in color_moods)
    overall_mood = mood_counts.most_common(1)[0][0] if mood_counts else "Neutral"
    return overall_mood

def print_color_moods(detected_colors):
    color_moods = get_color_moods()
    for color in detected_colors:
        print(f"Detected {color}: {color_moods.get(color, 'No mood found.')}")
    
    overall_mood = determine_overall_mood(detected_colors)
    print(f"Overall Mood: {overall_mood}")

if __name__ == "__main__":
    image_path = "images/TEST.jpg"  
    detected_colors = detect_blues(image_path)
    print_color_moods(detected_colors)