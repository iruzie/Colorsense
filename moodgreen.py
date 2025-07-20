import cv2
import numpy as np

def get_green_shades():
    """ Returns a dictionary of green shades with their RGB values. """
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

def detect_greens(image_path):
    """ Detects green shades in an image and matches them to the closest predefined shade. """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    green_shades = get_green_shades()
    detected_colors = set()
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]
        
        closest_shade = min(
            green_shades.keys(),
            key=lambda k: np.linalg.norm(np.array(avg_color) - np.array(green_shades[k]))
        )
        detected_colors.add(closest_shade)
    
    return detected_colors

def print_color_moods(detected_colors):
    """ Prints detected green shades with their associated moods. """
    color_moods = {
        "Bright Green": "Energizing, refreshing, and lively.",
        "Kelly Green": "Balanced, traditional, and natural.",
        "Lime Green": "Playful, youthful, and vibrant.",
        "Forest Green": "Grounded, natural, and soothing.",
        "Shamrock Green": "Lucky, cheerful, and positive.",
        "Emerald Green": "Elegant, luxurious, and rich.",
        "Olive Green": "Earthy, serious, and classic.",
        "Jungle Green": "Adventurous, deep, and calming.",
        "Moss Green": "Muted, organic, and peaceful.",
        "Neon Green": "Bold, exciting, and futuristic.",
        "Dark Green": "Sophisticated, deep, and resilient.",
        "Sea Green": "Tranquil, oceanic, and fresh.",
        "Spring Green": "Growth-oriented, lively, and youthful.",
        "Mint Green": "Cool, calming, and rejuvenating.",
        "Hunter Green": "Strong, authoritative, and grounded.",
        "Teal Green": "Balanced, refreshing, and creative.",
        "Fern Green": "Natural, earthy, and reliable.",
        "Army Green": "Tough, rugged, and neutral.",
        "Chartreuse": "Energetic, innovative, and unique.",
        "Pine Green": "Refreshing, deep, and stable."
    }
    
    overall_mood = []
    for color in detected_colors:
        mood = color_moods.get(color, "No specific mood found.")
        print(f"Detected {color}: {mood}")
        overall_mood.append(mood)
    
    print("\nOverall Mood:", " | ".join(overall_mood))

if __name__ == "__main__":
    image_path = "images/TEST.jpg"  
    detected_colors = detect_greens(image_path)
    print_color_moods(detected_colors)