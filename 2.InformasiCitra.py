import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image_info(input_path, figsize=(8, 6)):
    """
    Menampilkan citra dan informasi resolusi spasial serta tingkat keabuan
    Parameters:
        input_path: path file gambar input
        figsize: tuple (width, height) untuk ukuran figure dalam inch
    """
    # Baca gambar
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Baca sebagai grayscale
    if img is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar dari {input_path}")
    
    # Dapatkan resolusi spasial (M x N)
    height, width = img.shape
    resolution = f"{height} x {width}"
    
    # Dapatkan tingkat keabuan (L)
    # Untuk gambar 8-bit, L = jumlah level intensitas unik (0-255)
    unique_levels = len(np.unique(img))
    
    # Buat figure dengan ukuran yang ditentukan
    plt.figure(figsize=figsize)
    
    # Tampilkan gambar
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Tambahkan informasi sebagai teks di atas gambar
    plt.title(f"Resolusi Spasial (M x N): {resolution}\nTingkat Keabuan (L): {unique_levels}", fontsize=12)
    
    # Atur layout
    plt.tight_layout()
    
    # Tampilkan plot
    plt.show()

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path gambar Anda
    input_image = "img/testing.jpg"
    
    try:
        # Tampilkan citra dan informasinya
        display_image_info(input_image, figsize=(7, 5))
    except Exception as e:
        print(f"Error: {str(e)}")