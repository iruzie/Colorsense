import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

# Expanded color dictionary with more colors and their RGB values
COLOR_NAMES = {
    "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255),
    "Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (128, 128, 128),
    "Orange": (255, 165, 0), "Pink": (255, 192, 203), "Purple": (128, 0, 128),
    "Brown": (165, 42, 42), "Lime": (0, 255, 0), "Navy": (0, 0, 128),
    "Teal": (0, 128, 128), "Olive": (128, 128, 0), "Maroon": (128, 0, 0),
    "Silver": (192, 192, 192), "Gold": (255, 215, 0), "Beige": (245, 245, 220),
    "Coral": (255, 127, 80), "Turquoise": (64, 224, 208), "Indigo": (75, 0, 130),
    "Lavender": (230, 230, 250), "Chocolate": (210, 105, 30), "Salmon": (250, 128, 114),
    "Crimson": (220, 20, 60), "Orchid": (218, 112, 214), "Dark Green": (0, 100, 0),
    "Deep Pink": (255, 20, 147), "Light Blue": (173, 216, 230), "Sky Blue": (135, 206, 235),
    "Forest Green": (34, 139, 34), "Tomato": (255, 99, 71), "Dark Orange": (255, 140, 0),
    "Slate Gray": (112, 128, 144), "Sea Green": (46, 139, 87), "Midnight Blue": (25, 25, 112),
}

# Mood dictionary mapping colors to moods
COLOR_MOODS = {
    "Red": "Passion, Energy, Excitement",
    "Green": "Nature, Growth, Harmony",
    "Blue": "Calm, Trust, Serenity",
    "Yellow": "Happiness, Optimism, Creativity",
    "Cyan": "Refreshment, Clarity, Communication",
    "Magenta": "Imagination, Innovation, Spirituality",
    "Black": "Mystery, Elegance, Power",
    "White": "Purity, Simplicity, Cleanliness",
    "Gray": "Neutrality, Balance, Sophistication",
    "Orange": "Enthusiasm, Fun, Warmth",
    "Pink": "Love, Compassion, Playfulness",
    "Purple": "Royalty, Luxury, Ambition",
    "Brown": "Stability, Reliability, Comfort",
    "Lime": "Freshness, Zest, Vitality",
    "Navy": "Professionalism, Confidence, Authority",
    "Teal": "Sophistication, Healing, Protection",
    "Olive": "Peace, Wisdom, Resilience",
    "Maroon": "Strength, Courage, Passion",
    "Silver": "Modern, Futuristic, High-tech",
    "Gold": "Wealth, Success, Prosperity",
    "Beige": "Relaxation, Warmth, Neutrality",
    "Coral": "Warmth, Sociability, Approachability",
    "Turquoise": "Calmness, Clarity, Emotional Balance",
    "Indigo": "Intuition, Depth, Mystery",
    "Lavender": "Grace, Elegance, Femininity",
    "Chocolate": "Comfort, Reliability, Earthiness",
    "Salmon": "Warmth, Friendliness, Approachability",
    "Crimson": "Power, Love, Intensity",
    "Orchid": "Luxury, Mystery, Creativity",
    "Dark Green": "Prestige, Wealth, Stability",
    "Deep Pink": "Romance, Playfulness, Boldness",
    "Light Blue": "Peace, Tranquility, Softness",
    "Sky Blue": "Openness, Freedom, Inspiration",
    "Forest Green": "Nature, Freshness, Renewal",
    "Tomato": "Energy, Warmth, Vibrancy",
    "Dark Orange": "Adventure, Confidence, Success",
    "Slate Gray": "Formality, Professionalism, Neutrality",
    "Sea Green": "Refreshment, Healing, Renewal",
    "Midnight Blue": "Dignity, Intelligence, Authority",
}

# Function to find the closest color name
def get_closest_color(rgb):
    color_tree = KDTree(list(COLOR_NAMES.values()))
    _, idx = color_tree.query(rgb)
    return list(COLOR_NAMES.keys())[idx]

# Step 1: Load Image
image_path = "images/test5.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Step 2: Resize and Reshape Image for K-Means
image_resized = cv2.resize(image, (300, 300))
pixels = image_resized.reshape(-1, 3)  # Convert image to a list of pixels

# Step 3: Apply K-Means Clustering to Find Dominant Color
kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)  # 1 cluster for the most dominant color
kmeans.fit(pixels)
dominant_color = kmeans.cluster_centers_[0].astype(int)  # Get RGB values

# Convert to HEX
hex_color = "#{:02x}{:02x}{:02x}".format(*dominant_color)

# Get Color Name
color_name = get_closest_color(dominant_color)

# Get Mood
color_mood = COLOR_MOODS.get(color_name, "Mood not defined for this color")

# Step 4: Prepare Text
text1 = f"Dominant Color: {color_name}"
text2 = f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"
text3 = f"HEX: {hex_color}"
text4 = f"Mood: {color_mood}"

# Step 5: Draw a Background Box for Text
cv2.rectangle(image_resized, (5, 5), (280, 100), (0, 0, 0), -1)  # Black box for better visibility

# Step 6: Write Text on Image
font_scale = 0.5
font_thickness = 1
text_color = (255, 255, 255)  # White text

cv2.putText(image_resized, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
cv2.putText(image_resized, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
cv2.putText(image_resized, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
cv2.putText(image_resized, text4, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

# Step 7: Show Image
cv2.imshow("Dominant Color", cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV display
cv2.waitKey(0)
cv2.destroyAllWindows()