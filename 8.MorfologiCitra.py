import cv2
import numpy as np
import matplotlib.pyplot as plt

while True:
    print("\nMorfologi Citra")
    print("1. Pilih gambar")
    print("0. Exit")
    pilihan = input("> ")
    
    if pilihan == "0":
        print("Keluar dari program...")
        break
    
    elif pilihan == "1":
        gambar = input("Masukkan nama gambar: ")
        image_path = f'img/{gambar}'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Gambar tidak ditemukan di direktori: {image_path}")
            continue
        
        # Binerisasi gambar (jika belum biner)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Definisikan Structuring Element (SE), misalnya 3x3 square
        se = np.ones((3, 3), dtype=np.uint8)
        
        # Operasi Erosi, Dilasi, Opening dan Closing
        erosion = cv2.erode(binary_image, se, iterations=1)
        dilation = cv2.dilate(binary_image, se, iterations=1)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, se)
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, se)
        
        # Tampilkan hasil menggunakan Matplotlib
        plt.figure(figsize=(10, 7))
        
        # Daftar gambar dan judul
        images = [(binary_image, 'Gambar Asli'), (erosion, 'Erosi'), 
                (dilation, 'Dilasi'), (opening, 'Opening'), 
                (closing, 'Closing')]

        # Plot semua gambar dalam loop
        for i, (img, title) in enumerate(images, 1):
            plt.subplot(2, 3, i)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    else:
        print("Pilihan tidak valid, silakan masukkan 0 atau 1.")