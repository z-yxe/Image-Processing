import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Form 1: Transformasi Intensitas
def image_negative(img):
    return 255 - img

def log_transformation(img, c=1):
    img_float = img.astype(float)
    return np.clip(c * np.log(1 + img_float), 0, 255).astype(np.uint8)

def power_law_transformation(img, gamma, c=1):
    img_float = img.astype(float) / 255.0
    return np.clip(c * np.power(img_float, gamma) * 255, 0, 255).astype(np.uint8)

def piecewise_linear_transformation(img):
    img_float = img.astype(float)
    transformed = np.zeros_like(img_float)
    mask1 = img_float < 85
    mask2 = (img_float >= 85) & (img_float < 170)
    mask3 = img_float >= 170
    transformed[mask1] = img_float[mask1] * 0.5
    transformed[mask2] = (img_float[mask2] - 85) * 1.0 + 42.5
    transformed[mask3] = (img_float[mask3] - 170) * 1.5 + 127.5
    return np.clip(transformed, 0, 255).astype(np.uint8)

# Form 2: Bit-plane Slicing
def bit_plane_slicing(img, bit):
    return (img & (1 << bit)) * 255

# Form 3: Operasi Aritmatika
def image_subtraction(img1, img2):
    return cv2.absdiff(img1, img2)

# Form 4: Operasi Logika
def logical_and(img1, img2):
    return cv2.bitwise_and(img1, img2)

def logical_or(img1, img2):
    return cv2.bitwise_or(img1, img2)

def logical_xor(img1, img2):
    return cv2.bitwise_xor(img1, img2)

# Fungsi Penampil Hasil
def display_results(images, titles, grid_layout, figsize=(7, 5), window_title='Processing Results'):
    """Fungsi umum untuk menampilkan gambar dalam grid dengan loop"""
    plt.figure(figsize=figsize, num=window_title)
    rows, cols = grid_layout
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.tkraise()
    plt.show()

# Fungsi Utama
def process_and_display(input_path1, input_path2=None, form_choice=None):
    img = cv2.imread(input_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(input_path2, cv2.IMREAD_GRAYSCALE) if input_path2 else None
    if img is None:
        raise FileNotFoundError("Gambar 1 tidak ditemukan!")
    if form_choice in [3, 4] and img2 is None:
        raise FileNotFoundError("Gambar 2 diperlukan untuk Form 3 atau 4!")

    # Konfigurasi untuk setiap form, termasuk original image
    forms = {
        1: {
            'process': lambda: [
                img,
                image_negative(img),
                log_transformation(img, c=30),
                power_law_transformation(img, gamma=0.4),
                piecewise_linear_transformation(img)
            ],
            'titles': ['Original Image', 'Negative', 'Log (c=30)', 'Power-Law (γ=0.4)', 'Piecewise'],
            'grid': (2, 3),
            'figsize': (7, 5)
        },
        2: {
            'process': lambda: [img] + [bit_plane_slicing(img, bit) for bit in range(8)],
            'titles': ['Original Image'] + [f'Bit-plane {bit}' for bit in range(8)],
            'grid': (3, 3),
            'figsize': (7, 5)
        },
        3: {
            'process': lambda: [img, img2, image_subtraction(img, img2)],
            'titles': ['Original Image', 'Shifted Image', 'Subtraction'],
            'grid': (1, 3),
            'figsize': (7, 5)
        },
        4: {
            'process': lambda: [img, img2, logical_and(img, img2), logical_or(img, img2), logical_xor(img, img2)],
            'titles': ['Original Image', 'Shifted Image', 'AND', 'OR', 'XOR'],
            'grid': (2, 3),
            'figsize': (7, 5)
        }
    }

    if form_choice not in forms:
        return False

    # Proses dan tampilkan hasil
    config = forms[form_choice]
    results = config['process']()
    display_results(results, config['titles'], config['grid'], figsize=config['figsize'], 
                    window_title=f'Form {form_choice} Results')
    return True

# Main
if __name__ == "__main__":
    input_image1 = "img/1.jpg"
    input_image2 = "img/2.jpg"

    print("Pilih form yang ingin diproses:")
    print("1. Transformasi Intensitas")
    print("2. Bit-plane Slicing")
    print("3. Operasi Aritmatika (Subtraction)")
    print("4. Operasi Logika")

    while True:
        try:
            form_choice = int(input("> "))
            if form_choice == 0:
                print("Keluar dari program.")
                break
            if not process_and_display(input_image1, input_image2, form_choice):
                print("Pilihan tidak valid. Masukkan angka antara 0-4.")
        except ValueError:
            print("Masukkan angka yang valid!")
        except Exception as e:
            print(f"Error: {str(e)}")