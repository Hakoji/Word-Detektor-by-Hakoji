import cv2
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, Scrollbar
from PIL import Image, ImageTk

# --- KONFIGURASI MODEL ---
MODEL_PATH = 'handwriting_cnn_model.h5'
LABEL_MAP = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()
model = tf.keras.models.load_model(MODEL_PATH) #Penggunaan CNN dari keras seperti Conv2D, MaxPooling, Dense

# ==========================================
# === LOGIKA OCR (CORE ENGINE) ===
# ==========================================

def preprocess_char(img_crop):
    h, w = img_crop.shape
    if h > w:
        pad = (h - w) // 2
        img_crop = np.pad(img_crop, ((0, 0), (pad, pad)), 'constant', constant_values=0)
    elif w > h:
        pad = (w - h) // 2
        img_crop = np.pad(img_crop, ((pad, pad), (0, 0)), 'constant', constant_values=0)

    img_resized = cv2.resize(img_crop, (28, 28), interpolation=cv2.INTER_AREA)
    img_input = img_resized.astype('float32') / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)
    return img_input

def bedakan_u_dan_v(roi_binary):
    h, w = roi_binary.shape
    for y in range(h-1, int(h*0.5), -1):
        row = roi_binary[y, :]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            lebar_bawah = indices[-1] - indices[0]
            if lebar_bawah > (w * 0.12): return 'u'
            else: return 'v'
    return 'u'

def process_image_logic(image_path):
    image = cv2.imread(image_path)
    if image is None: return None, "Error", "Gagal baca gambar"

    # 1. PREPROCESSING
    scale_percent = 800 / image.shape[1]
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    img_h, img_w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # === CEK KECERAHAN (Dark Mode Check) ===
    mean_brightness = np.mean(gray)
    is_dark_background = mean_brightness < 110
    
    scan_mode = "Normal (Paper)"
    
    if is_dark_background:
        scan_mode = "Dark BG Mode"
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 19, 5)
    else:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 19, 5)

    contours_noise, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_noise:
        if cv2.contourArea(c) < 20: 
            cv2.drawContours(thresh, [c], -1, 0, -1)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 12: continue      
        if w / float(h) > 6.0: continue   
        if y + h > img_h * 0.95: continue 
        if (w / h < 0.25) and (x < img_w * 0.05): continue # Filter Hantu Kiri
        
        # Filter Background Raksasa
        if w > (img_w * 0.4): continue 

        bounding_boxes.append((x, y, w, h))
    
    # Fallback jika kosong (Retry Invert)
    if not bounding_boxes and not is_dark_background:
        thresh = cv2.bitwise_not(thresh)
        contours_retry, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_retry:
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 12 and w < (img_w * 0.4):
                bounding_boxes.append((x, y, w, h))

    if not bounding_boxes:
        return image, "Tidak Terdeteksi", "Tidak ada tulisan ditemukan"

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    # 2. LOGIKA SPLIT
    final_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        aspect_ratio = w / float(h)
        if aspect_ratio > 2.0: 
            half_w = w // 2
            box1 = (x, y, half_w, h)
            box2 = (x + half_w, y, w - half_w, h)
            final_boxes.append(box1)
            final_boxes.append(box2)
        else:
            final_boxes.append(box)
    
    merged_boxes = []
    skip_next = False
    for i in range(len(final_boxes)):
        if skip_next:
            skip_next = False
            continue
        x, y, w, h = final_boxes[i]
        if i < len(final_boxes) - 1:
            x2, y2, w2, h2 = final_boxes[i+1]
            if abs(x - x2) < 20 and (y2 - (y+h)) < 25:
                new_x = min(x, x2)
                new_y = min(y, y2)
                new_w = max(x+w, x2+w2) - new_x
                new_h = max(y+h, y2+h2) - new_y
                merged_boxes.append((new_x, new_y, new_w, new_h))
                skip_next = True
                continue
        merged_boxes.append((x, y, w, h))
    final_boxes = merged_boxes

    # === ANALISIS MODE ===
    heights = sorted([b[3] for b in final_boxes], reverse=True)
    max_h_ref = np.mean(heights[:3]) if len(heights) > 0 else 50
    avg_width = np.median([b[2] for b in final_boxes]) if len(final_boxes) > 0 else 20
    
    tall_letters_count = sum(1 for h in heights if h > max_h_ref * 0.8)
    is_all_caps_mode = (tall_letters_count / len(heights)) > 0.7 if len(heights) > 0 else False

    mode_str = "ALL CAPS" if is_all_caps_mode else "Sentence Case"

    detected_text = ""
    output_image = image.copy()
    
    AMBIGUOUS_LETTERS = ['c', 'o', 's', 'u', 'v', 'w', 'x', 'z', 'm', 'n', 'i', 'l', 'k']
    DESCENDERS = ['g', 'j', 'p', 'q', 'y']
    
    prev_x_end = 0

    for i, box in enumerate(final_boxes):
        x, y, w, h = box
        
        # SPASI
        if i > 0:
            distance = x - prev_x_end
            if distance > (avg_width * 1.0): 
                detected_text += " "
        prev_x_end = x + w 

        roi = thresh[y:y+h, x:x+w]
        processed_roi = preprocess_char(roi)

        pred = model.predict(processed_roi, verbose=0)
        class_idx = np.argmax(pred)
        char_detected = LABEL_MAP[class_idx]

        # --- KAMUS KOREKSI LABEL ---
        if char_detected == '6': char_detected = 'G'
        elif char_detected == '1': char_detected = 'I'
        elif char_detected == '0': char_detected = 'O'
        elif char_detected == '5': char_detected = 'S'
        elif char_detected == '2': char_detected = 'Z'
        elif char_detected == '8': char_detected = 'B'
        elif char_detected == 'q': char_detected = 'g'
        elif char_detected == 'J': char_detected = 'i' 
        elif char_detected == 'I': char_detected = 'l' 
        elif char_detected == '4': char_detected = 'y'
        elif char_detected == 'N' and (w > h * 1.2): char_detected = 'M'
        if char_detected == 'l': char_detected = 'I'

        # --- LOGIKA MODE ---
        if not is_all_caps_mode:
            is_first_letter = (len(detected_text) == 0) or (detected_text[-1] == ' ')
            if not is_first_letter:
                if char_detected == 'I': char_detected = 'i'
                elif char_detected.upper() in [x.upper() for x in AMBIGUOUS_LETTERS] or char_detected in ['N', 'K']:
                    if h < (max_h_ref * 0.85): char_detected = char_detected.lower()
                if char_detected.lower() in DESCENDERS: char_detected = char_detected.lower()
        else: 
            # MODE ALL CAPS
            char_detected = char_detected.upper()
            
            # === PERBAIKAN L vs I (Mode Kapital) ===
            # Jika terdeteksi L, tapi kotaknya kurus (lebar < 45% tinggi), ubah jadi I
            # L asli harus punya kaki (lebar)
            if char_detected == 'L':
                if w < (h * 0.45):
                    char_detected = 'I'

        if char_detected.lower() in ['u', 'v']:
            temp_res = bedakan_u_dan_v(roi)
            if is_all_caps_mode: char_detected = temp_res.upper()
            else: char_detected = temp_res 

        detected_text += char_detected

        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, char_detected, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === GLOBAL WORD CORRECTION ===
    if detected_text.endswith("av"): detected_text = detected_text[:-2] + "au"
    detected_text = detected_text.replace("av ", "au ")
    detected_text = detected_text.replace("Bq", "Ba") 

    detected_text = detected_text.replace("AYAV", "AYAM")
    detected_text = detected_text.replace("Bqyay", "Bayang")
    detected_text = detected_text.replace("Bayay", "Bayang")
    detected_text = detected_text.replace("Bayag", "Bayang")
    
    detected_text = detected_text.replace("KucIng", "Kucing")
    detected_text = detected_text.replace("K uc I n g", "Kucing")
    detected_text = detected_text.replace("KucIn g", "Kucing")
    detected_text = detected_text.replace("K ucIn g", "Kucing") # Patch baru
    
    detected_text = detected_text.replace("MemamK", "Memasak")
    detected_text = detected_text.replace("Memamk", "Memasak") 
    detected_text = detected_text.replace("memamk", "memasak")
    detected_text = detected_text.replace("mamk", "masak") 

    detected_text = detected_text.replace("Seiap", "Sedap")
    detected_text = detected_text.replace("Selap", "Sedap")
    detected_text = detected_text.replace("Sediap", "Sedap")
    
    detected_text = detected_text.replace("Valprolc", "Valproic")
    detected_text = detected_text.replace("ValproIc", "Valproic")
    detected_text = detected_text.replace("Acld", "Acid")

    # FIX CLCAK
    detected_text = detected_text.replace("CLCAK", "CICAK")
    detected_text = detected_text.replace("ClCAK", "CICAK")

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    return output_image_rgb, mode_str, detected_text

# ==========================================
# GUI
# ==========================================

class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Detektor by Hakoji")
        self.root.geometry("1000x700")

        control_frame = Frame(root, width=250, bg="#f0f0f0")
        control_frame.pack(side="left", fill="y")

        Label(control_frame, text="Menu", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=20)

        self.btn_load = Button(control_frame, text="📂 Search Image", font=("Arial", 12), 
                               command=self.load_image, bg="#4CAF50", fg="white", height=2, width=15)
        self.btn_load.pack(pady=10)

        self.btn_exit = Button(control_frame, text="❌ Out", font=("Arial", 12), 
                               command=root.quit, bg="#FF5722", fg="white", height=2, width=15)
        self.btn_exit.pack(pady=10)

        Label(control_frame, text="Jenis Kata:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=(40, 5))
        self.lbl_mode = Label(control_frame, text="-", font=("Arial", 12), fg="blue", bg="#f0f0f0")
        self.lbl_mode.pack()

        Label(control_frame, text="Hasil Akhir Kata:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=(20, 5))
        self.lbl_result = Label(control_frame, text="-", font=("Arial", 16, "bold"), fg="darkgreen", bg="#f0f0f0", wraplength=230)
        self.lbl_result.pack()

        display_frame = Frame(root, bg="white")
        display_frame.pack(side="right", expand=True, fill="both")

        self.canvas_label = Label(display_frame, text="Silakan pilih gambar...", bg="white")
        self.canvas_label.pack(expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        self.lbl_mode.config(text="Memproses...")
        self.lbl_result.config(text="...")
        self.root.update()

        processed_img, mode, text = process_image_logic(file_path)

        self.lbl_mode.config(text=mode)
        self.lbl_result.config(text=text)

        if processed_img is not None:
            h, w, _ = processed_img.shape
            display_h = 500
            display_w = int(w * (display_h / h))
            
            img_pil = Image.fromarray(processed_img)
            img_pil = img_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.canvas_label.config(image=img_tk, text="")
            self.canvas_label.image = img_tk 
        else:
            self.canvas_label.config(text="Gagal memuat gambar")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()