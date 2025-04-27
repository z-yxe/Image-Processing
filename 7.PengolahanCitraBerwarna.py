import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    """Class for handling image processing operations"""
    
    def __init__(self, image_path=None):
        """Initialize with optional image path"""
        self.image = None
        self.image_path = None
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load an image and validate it has correct color channels"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"Failed to load image at: {image_path}")
            
        # Validasi format warna (harus BGR)
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            print("Warning: Image is not in expected BGR format")
            
        self.image_path = image_path
        return self.image
    
    def display_image(self, title, image=None, channels=None):
        if image is None:
            image = self.image
            
        # PENTING: Selalu konversi dari BGR ke RGB untuk visualisasi
        if image is not None and image.ndim == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
            
        plt.figure(figsize=(15, 5) if channels else (8, 6))
        
        if channels:
            for i, (ch_name, ch_data) in enumerate(channels.items(), 1):
                plt.subplot(1, len(channels), i)
                plt.title(ch_name)
                
                # Pastikan setiap channel ditampilkan dengan benar
                if ch_name == 'Original' or 'Sharpened' or 'Smoothing' in ch_name:
                    # Gambar berwarna perlu dikonversi BGR->RGB
                    if ch_data.ndim == 3 and ch_data.shape[2] == 3:
                        plt.imshow(cv2.cvtColor(ch_data, cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(ch_data, cmap='gray')
                elif ch_data.ndim == 2:  # Kanal grayscale
                    plt.imshow(ch_data, cmap='gray')
                else:  # Kanal berwarna
                    plt.imshow(ch_data)
                
                plt.axis('off')
            plt.tight_layout()
        else:
            plt.title(title)
            plt.imshow(display_image)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def rgb_to_cmy(self):
        """Convert RGB to CMY"""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to CMY using formula from the PDF: [C,M,Y] = [255,255,255] - [R,G,B]
        cmy = 255 - rgb_img
        cmy_img = cmy.astype(np.uint8)
        
        # Split channels for visualization
        c, m, y = cv2.split(cmy_img)
        
        channels = {
            'Cyan': c,
            'Magenta': m,
            'Yellow': y,
            'CMY' : cmy_img
        }
        
        return cmy_img, channels

    def rgb_to_cmyk(self):
        """Convert RGB to CMYK"""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Normalize RGB to [0,1]
        rgb = rgb_img.astype(float) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Calculate Black (K) as per "four-color printing" approach
        k = np.minimum.reduce([1 - r, 1 - g, 1 - b])
        
        # Calculate CMY using K
        # Avoid division by zero where k=1
        mask = k < 0.9999  # Use a threshold close to 1
        
        c = np.zeros_like(k)
        m = np.zeros_like(k)
        y = np.zeros_like(k)
        
        c[mask] = (1 - r[mask] - k[mask]) / (1 - k[mask])
        m[mask] = (1 - g[mask] - k[mask]) / (1 - k[mask])
        y[mask] = (1 - b[mask] - k[mask]) / (1 - k[mask])
        
        # Scale back to [0,255]
        c_255 = (c * 255).astype(np.uint8)
        m_255 = (m * 255).astype(np.uint8)
        y_255 = (y * 255).astype(np.uint8)
        k_255 = (k * 255).astype(np.uint8)
        
        # Create merged CMYK image (visualization only)
        cmyk_img = cv2.merge([c_255, m_255, y_255, k_255])
        
        # Create channel visualizations
        channels = {
            'Cyan': c_255,
            'Magenta': m_255,
            'Yellow': y_255,
            'Black': k_255,
            'CMYK' : cmyk_img
        }
        
        return cmyk_img, channels

    def rgb_to_hsi(self):
        """Convert RGB to HSI (Hue, Saturation, Intensity) using the formulas from the PDF"""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Normalize RGB to [0,1]
        rgb = rgb_img.astype(float) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Calculate Intensity as per PDF: I = (R+G+B)/3
        i = (r + g + b) / 3.0
        
        # Calculate Saturation using the formula from PDF: S = 1 - 3*min(R,G,B)/(R+G+B)
        min_rgb = np.minimum.reduce([r, g, b])
        s = np.zeros_like(i)
        # Avoid division by zero
        rgb_sum = r + g + b
        valid = rgb_sum > 0
        s[valid] = 1 - (3 * min_rgb[valid]) / (rgb_sum[valid])
        s = np.clip(s, 0, 1)  # Ensure saturation is between 0 and 1
        
        # Calculate Hue using the exact formula from the PDF
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b) * (g - b) + 1e-6)  # Avoid division by zero
        theta = np.arccos(np.clip(num / den, -1.0, 1.0))
        
        h = theta.copy()
        h[b > g] = 2 * np.pi - h[b > g]
        
        # Normalize hue to [0,1]
        h = h / (2 * np.pi)
        
        # Scale to uint8 range
        h_255 = (h * 255).astype(np.uint8)
        s_255 = (s * 255).astype(np.uint8)
        i_255 = (i * 255).astype(np.uint8)
        
        # Create channel visualizations (with colormap for hue)
        h_colored = cv2.applyColorMap(h_255, cv2.COLORMAP_HSV)
        
        # Create merged HSI image
        hsi_img = cv2.merge([h_255, s_255, i_255])
        
        channels = {
            'Hue': h_colored,
            'Saturation': s_255,
            'Intensity': i_255,
            'HSI' : hsi_img
        }
        
        return hsi_img, channels

    def rgb_to_yuv(self):
        """Convert RGB to YUV using the Gonzales (2002) formula from the PDF"""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Normalize RGB to [0,1]
        rgb = rgb_img.astype(float) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Implementing Gonzales (2002) formula as mentioned in the PDF
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.169 * r - 0.331 * g + 0.500 * b
        v = 0.500 * r - 0.419 * g - 0.081 * b
        
        # Normalize and convert to uint8
        # YUV typically has Y in [0, 1] and U,V in [-0.5, 0.5], need to adjust for visualization
        y_255 = (y * 255).astype(np.uint8)
        u_scaled = ((u + 0.5) * 255).astype(np.uint8)  # Shift from [-0.5, 0.5] to [0, 1]
        v_scaled = ((v + 0.5) * 255).astype(np.uint8)  # Shift from [-0.5, 0.5] to [0, 1]
        
        # Merge channels for YUV representation
        yuv_img = cv2.merge([y_255, u_scaled, v_scaled])
        
        # Create visualizations for each channel
        channels = {
            'Y (Luminance)': y_255,
            'U (Chrominance)': u_scaled,
            'V (Chrominance)': v_scaled,
            'YUV' : yuv_img
        }
        
        return yuv_img, channels

    def rgb_to_ycbcr(self):
        """Convert RGB to YCbCr using the Tarek M (2008) formula from the PDF"""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Normalize RGB to [0,1]
        rgb = rgb_img.astype(float) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Using Tarek M (2008) formula as mentioned in the PDF
        y = 16 + (65.4810 * r + 128.5530 * g + 24.9660 * b)
        cr = 128 + (-37.7745 * r - 74.1529 * g + 111.9337 * b)
        cb = 128 + (111.9581 * r - 93.7509 * g - 18.2072 * b)
        
        # Clip values to [0, 255] range and convert to uint8
        y = np.clip(y, 0, 255).astype(np.uint8)
        cr = np.clip(cr, 0, 255).astype(np.uint8)
        cb = np.clip(cb, 0, 255).astype(np.uint8)
        
        # Create merged YCbCr image
        ycbcr_img = cv2.merge([y, cb, cr])
        
        # Create visualizations for each channel
        channels = {
            'Y (Luminance)': y,
            'Cb (Blue)': cb,
            'Cr (Red)': cr,
            'YCbCr' : ycbcr_img
        }
        
        return ycbcr_img, channels  
 
    def smooth_image(self, method, params):
        """Apply smoothing filter to the image (manual mean sesuai konteks PDF)"""
        if self.image is None:
            raise ValueError("No image loaded")

        if method == 'Mean':
            ksize = self._validate_kernel_size(params.get('ksize', 3))
            pad = ksize // 2

            # Pisah channel
            b, g, r = cv2.split(self.image)

            # Fungsi untuk filter mean manual
            def apply_mean_filter(channel):
                return cv2.blur(channel, (ksize, ksize))

            b_filtered = apply_mean_filter(b)
            g_filtered = apply_mean_filter(g)
            r_filtered = apply_mean_filter(r)

            smoothing = cv2.merge((b_filtered, g_filtered, r_filtered))
            difference = self.image - smoothing
            title = "Mean Filter"

            channels = {
                'Original': self.image,
                'Smoothing': smoothing,
                'Difference': difference
            }
            self.display_image("Mean Filter", None, channels)
            return smoothing, title

        # Metode lainnya tetap
        return super().smooth_image(method, params)

    def sharpen_image(self, method, params):
        """Apply sharpening using Laplacian sesuai PDF: per channel Laplacian"""
        if self.image is None:
            raise ValueError("No image loaded")

        if method == 'Laplacian':
            ksize = self._validate_kernel_size(params.get('ksize', 3), allow_one=True)

            # Pisahkan channel
            b, g, r = cv2.split(self.image)

            # Terapkan Laplacian ke masing-masing channel
            def laplacian_sharpen(channel):
                lap = cv2.Laplacian(channel, cv2.CV_64F, ksize=ksize)
                lap_abs = cv2.convertScaleAbs(lap)
                return cv2.addWeighted(channel, 1.0, lap_abs, 1.0, 0)

            b_sharp = laplacian_sharpen(b)
            g_sharp = laplacian_sharpen(g)
            r_sharp = laplacian_sharpen(r)

            sharpened = cv2.merge((b_sharp, g_sharp, r_sharp))

            # Hitung perbedaan antara original dan hasil sharpened
            difference = self.image - sharpened

            title = "Laplacian Sharpening"

            channels = {
                'Original': self.image,
                'Sharpened': sharpened,
                'Difference': difference
            }
            self.display_image("Laplacian RGB Per-Channel", None, channels)
            return sharpened, title

        # Metode lainnya tetap
        return super().sharpen_image(method, params)

    def _validate_kernel_size(self, ksize, allow_one=False):
        if ksize > 1 and ksize % 2 == 0:
            ksize += 1  # Make it odd
        return ksize
    
    def _display_filter_result(self, title, original, result):
        """Display original and filtered images side by side"""
        plt.figure(figsize=(12, 6))
        
        # Konversi BGR ke RGB untuk visualisasi yang benar
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(original_rgb)  # Pastikan ini RGB
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(title)
        plt.imshow(result_rgb)  # Pastikan ini RGB
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def get_user_input(prompt, input_type=str, min_value=None, max_value=None, default=None):
    """Get and validate user input with specified constraints"""
    prompt_with_default = f"{prompt} [{default}]: " if default is not None else prompt
    
    while True:
        try:
            user_input = input(prompt_with_default).strip()
            
            # Use default if input is empty
            if not user_input and default is not None:
                return default
                
            # Convert to the specified type
            value = input_type(user_input)
            
            # Validate range if specified
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
                
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
                
            return value
            
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")


def main():
    image_path = "img/forest.jpg"
    processor = ImageProcessor(image_path)

    color_conversions = {
        '1': ('CMY', processor.rgb_to_cmy),
        '2': ('CMYK', processor.rgb_to_cmyk),
        '3': ('HSI', processor.rgb_to_hsi),
        '4': ('YUV', processor.rgb_to_yuv),
        '5': ('YCbCr', processor.rgb_to_ycbcr)
    }

    while True:
        print("\n===== COLOR IMAGE PROCESSING APPLICATION =====")
        print("1. Konversi warna")
        print("2. Smoothing (Mean Filter)")
        print("3. Sharpening (Laplacian)")
        print("0. Keluar")
        choice = input("Pilih opsi: ")

        if choice == '1':
            print("\nPilih konversi warna:")
            for k, (name, _) in color_conversions.items():
                print(f"{k}. {name}")
            sub_choice = input("Pilihan: ")
            if sub_choice in color_conversions:
                _, func = color_conversions[sub_choice]
                img, channels = func()
                processor.display_image(f"Konversi Warna - {color_conversions[sub_choice][0]}", None, channels)

        elif choice == '2':
            ksize = get_user_input("Kernel size untuk smoothing (ganjil, e.g. 3, 5, 7)", int, 3, 55, 5)
            processor.smooth_image('Mean', {'ksize': ksize})

        elif choice == '3':
            ksize = get_user_input("Kernel size untuk sharpening (1, 3, 5, 7)", int, 1, 7, 3)
            processor.sharpen_image('Laplacian', {'ksize': ksize})

        elif choice == '0':
            print("Terima kasih!")
            break

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()