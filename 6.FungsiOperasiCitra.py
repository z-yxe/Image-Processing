import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Histogram Equalization
def histogram_equalization(img):
    """Melakukan ekualisasi histogram"""
    return cv2.equalizeHist(img)

# Spatial Smoothing
def spatial_smoothing(img, kernel_size=3):
    """Melakukan semua metode smoothing spasial"""
    avg = cv2.blur(img, (kernel_size, kernel_size))
    median = cv2.medianBlur(img, kernel_size)
    gauss = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return avg, gauss, median

# Spatial Sharpening
def spatial_sharpening(img):
    """Melakukan metode sharpening spasial"""
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = img - laplacian
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    return laplacian, unsharp

# Penampil Hasil Tkinter
def display_results(images, titles, grid_layout, figsize=(7, 5), window_title='Image Processing Results'):
    """Fungsi umum untuk menampilkan gambar dalam grid dengan loop for"""
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
def process_image(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Gambar tidak ditemukan!")
    
    while True:
        print("\nPilih fungsi pengolahan citra:")
        print("1. Histogram Equalization")
        print("2. Spatial Smoothing")
        print("3. Spatial Sharpening")
        
        try:
            choice = int(input("> "))
            
            if choice == 0:
                print("Keluar dari program.")
                break
            
            if choice == 1:
                result = histogram_equalization(img)
                images = [img, result]
                titles = ['Original Image', 'Histogram Equalization']
                grid_layout = (1, 2)
                display_results(images, titles, grid_layout, figsize=(7, 5), 
                              window_title='Histogram Equalization Results')
            
            elif choice == 2:
                avg, gauss, median = spatial_smoothing(img)
                images = [img, avg, gauss, median]
                titles = ['Original Image', 'Smoothing - Average', 'Smoothing - Gaussian', 'Smoothing - Median']
                grid_layout = (2, 2)
                display_results(images, titles, grid_layout, figsize=(7, 5), 
                              window_title='Spatial Smoothing Results')
            
            elif choice == 3:
                laplacian, unsharp = spatial_sharpening(img)
                images = [laplacian, img, unsharp]
                titles = ['Sharpening - Laplacian', 'Original Image', 'Sharpening - Unsharp']
                grid_layout = (1, 3)
                display_results(images, titles, grid_layout, figsize=(7, 5), 
                              window_title='Spatial Sharpening Results')
            
            else:
                print("Pilihan tidak valid. Masukkan angka antara 0-3.")
        
        except ValueError:
            print("Masukkan angka yang valid!")
        except Exception as e:
            print(f"Error: {str(e)}")

# Main
if __name__ == "__main__":
    input_image = "img/eagle.jpg"
    
    try:
        process_image(input_image)
    except Exception as e:
        print(f"Error: {str(e)}")