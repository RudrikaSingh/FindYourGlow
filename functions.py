import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import face_recognition

def get_dominant_color(image, k=4, image_processing_size=None):
    """
    Returns the dominant color of an image using KMeans clustering.

    :param image: The input image
    :param k: Number of clusters
    :param image_processing_size: Resize image for faster processing
    :return: Tuple of the dominant color in BGR format
    """
    if image_processing_size:
        image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)
    
    # Reshape the image to be a list of pixels
    pixels = np.float32(image.reshape(-1, 3))

    # KMeans clustering to find dominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the most dominant cluster
    counts = Counter(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]

    return tuple(dominant_color.astype(int))

def analyze_image(image_path):
    """
    Analyze the image to find the dominant color in the facial region.

    :param image_path: Path to the image
    :return: Dominant color in the facial region
    """
    image = cv2.imread(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        raise ValueError("No faces found in the image.")
    
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    
    dominant_color = get_dominant_color(face_image)
    
    return dominant_color
