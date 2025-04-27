import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi konversi RGB ke Grayscale
def rgb_to_gray_lightness(img):
    """Konversi RGB ke Gray menggunakan Lightness Method"""
    return (np.max(img, axis=2) + np.min(img, axis=2)) / 2

def rgb_to_gray_average(img):
    """Konversi RGB ke Gray menggunakan Average Method"""
    return np.mean(img, axis=2)

def rgb_to_gray_luminosity(img):
    """Konversi RGB ke Gray menggunakan Luminosity Method"""
    return np.dot(img[..., :3], [0.21, 0.71, 0.07])

def convert_and_display_gray(input_path, figsize=(4, 5)):
    """Konversi gambar RGB ke Grayscale dan tampilkan menggunakan Matplotlib"""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar dari {input_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Daftar metode konversi dan judul
    methods = [
        ('Original', lambda x: x, False),
        ('Lightness', rgb_to_gray_lightness, True),
        ('Original', lambda x: x, False),
        ('Average', rgb_to_gray_average, True),
        ('Original', lambda x: x, False),
        ('Luminosity', rgb_to_gray_luminosity, True)
    ]

    # Proses konversi
    images = [method[1](img_rgb) if method[0] != 'Original' else img_rgb 
              for method in methods]
    images = [img.astype(np.uint8) if method[2] else img 
              for img, method in zip(images, methods)]

    # Tampilkan semua gambar dalam satu figure
    plt.figure(figsize=figsize)
    for i, (title, _, use_gray) in enumerate(methods, 1):
        plt.subplot(3, 2, i)
        plt.imshow(images[i-1], cmap='gray' if use_gray else None)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    input_image = "img/forest.jpg"
    convert_and_display_gray(input_image)