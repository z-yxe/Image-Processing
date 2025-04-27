import cv2
import numpy as np

def select_roi_and_get_matrix(input_path, window_size=(700, 500)):
    """
    Memilih bagian gambar dan mengubahnya menjadi matriks piksel dengan ukuran jendela tetap
    Parameters:
        input_path: path file gambar input
        window_size: tuple (width, height) dalam piksel untuk ukuran jendela
    """
    # Baca gambar sebagai grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar dari {input_path}")
    
    # Tentukan ukuran jendela tetap (misalnya 700x500 piksel)
    display_width, display_height = window_size
    
    # Skalakan gambar agar sesuai dengan ukuran jendela tetap sambil mempertahankan rasio aspek
    orig_height, orig_width = img.shape
    scale = min(display_width / orig_width, display_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Tampilkan gambar untuk pemilihan ROI
    roi = cv2.selectROI("Select ROI", resized_img, fromCenter=False, showCrosshair=True)
    
    # Tutup jendela setelah ROI dipilih
    cv2.destroyAllWindows()
    
    # Dapatkan koordinat ROI dari gambar yang diskalakan
    x, y, w, h = roi
    
    # Jika ROI tidak valid (misalnya tidak ada wilayah yang dipilih), keluar
    if w == 0 or h == 0:
        print("Pemilihan ROI dibatalkan atau tidak valid.")
        return None
    
    # Konversi koordinat ROI dari gambar yang diskalakan ke gambar asli
    x_orig = int(x / scale)
    y_orig = int(y / scale)
    w_orig = int(w / scale)
    h_orig = int(h / scale)
    
    # Potong gambar asli berdasarkan ROI yang disesuaikan
    roi_img = img[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
    
    # Matriks piksel adalah ROI yang dipotong
    pixel_matrix = roi_img
    
    # Tampilkan informasi
    print(f"\nResolusi Spasial ROI (M x N): {h_orig} x {w_orig}")
    print(f"Tipe Data Matriks: {pixel_matrix.dtype}")
    print(f"Ukuran Matriks: {pixel_matrix.shape}")
    print("\nMatriks Piksel ROI:")
    print(pixel_matrix)
    
    # Tampilkan ROI yang dipilih
    cv2.imshow("Selected ROI", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Kembalikan matriks piksel
    return pixel_matrix

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path gambar Anda
    input_image = "img/testing.jpg"
    
    try:
        # Pilih ROI dan dapatkan matriks piksel dengan ukuran jendela tetap
        matrix = select_roi_and_get_matrix(input_image, window_size=(700, 500))
        
        if matrix is not None:
            print("Matriks piksel berhasil dibuat!")
    except Exception as e:
        print(f"Error: {str(e)}")