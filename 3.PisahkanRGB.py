import cv2
import numpy as np
import matplotlib.pyplot as plt

def separate_rgb_channels(input_path):
    """Pisahkan channel RGB dari gambar"""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError("Gambar tidak ditemukan!")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return {
        'Original Image': (img_rgb, False),  # (gambar, apakah grayscale)
        'Red Channel': (img_rgb[:, :, 0], True),
        'Green Channel': (img_rgb[:, :, 1], True),
        'Blue Channel': (img_rgb[:, :, 2], True)
    }

def display_rgb_channels(input_path, window_size=(7, 5)):
    """
    Tampilkan gambar asli dan channel RGB dalam satu jendela
    Parameters:
        input_path: path file gambar input
        window_size: tuple (width, height) dalam inch
    """
    # Dapatkan gambar dan channel
    channels = separate_rgb_channels(input_path)
    
    # Buat figure dengan ukuran yang ditentukan
    plt.figure(figsize=window_size)
    
    # Tampilkan semua channel dengan loop
    for i, (title, (img, is_gray)) in enumerate(channels.items(), 1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap='gray' if is_gray else None)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Contoh penggunaan
if __name__ == "__main__":
    input_image = "img/forest.jpg"
    try:
        display_rgb_channels(input_image)
    except Exception as e:
        print(f"Error: {str(e)}")