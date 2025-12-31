import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import threading
import pygame

# --- CẤU HÌNH ---
SR = 22050
CLASSES = ["cel", "cla", "flu", "gac"]

# Khởi tạo pygame mixer cho phát nhạc
pygame.mixer.init()

# --- BƯỚC 1 & 2: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG ---
def extract_all_features(y, sr):
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    centroid = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
    cent_mean = np.mean(centroid)
    cent_std = np.std(centroid)
    zcr = librosa.feature.zero_crossing_rate(y_trim)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    features = (
        list(mfcc_mean) + list(mfcc_std) + [cent_mean, cent_std, zcr_mean, zcr_std]
    )
    return features

# --- BƯỚC 3: THUẬT TOÁN PHÂN LOẠI (LAZY LEARNING) ---
def lazy_classify(test_features, train_csv="data/nsynth_features_detail.csv"):
    df = pd.read_csv(train_csv)
    X_train = df.iloc[:, :-2].values
    y_train = df["family"].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform([test_features])
    distances = [distance.euclidean(test_scaled[0], x) for x in X_train_scaled]
    nearest_index = np.argmin(distances)
    return y_train[nearest_index]

# --- TÍNH NĂNG NÂNG CAO: TÁCH NGUỒN NMF ---
def separate_nmf(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    S = np.abs(librosa.stft(y))
    W, H = librosa.decompose.decompose(S, n_components=2, sort=True)
    output_files = []
    for i in range(2):
        S_i = np.outer(W[:, i], H[i, :])
        y_out = librosa.istft(S_i * np.exp(1j * np.angle(librosa.stft(y))))
        out_name = f"extracted_part_{i+1}.wav"
        sf.write(out_name, y_out, sr)
        output_files.append(out_name)
    return output_files

# --- GIAO DIỆN HIỆN ĐẠI VỚI NHIỀU FILE ---
class ModernMusicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Classification & Source Separation")
        self.root.geometry("900x700")
        self.root.configure(bg="#f5f5f5")
        
        self.audio_files = []  # Danh sách file đã tải
        self.current_playing = None  # File đang phát
        
        # Style configuration
        self.setup_styles()
        
        # Main container
        main_container = tk.Frame(root, bg="#f5f5f5")
        main_container.pack(fill="both", expand=True)
        
        # Left panel - File list
        left_panel = tk.Frame(main_container, bg="#f5f5f5", width=350)
        left_panel.pack(side="left", fill="both", expand=False, padx=(20, 10), pady=20)
        left_panel.pack_propagate(False)
        
        # Right panel - Actions and results
        right_panel = tk.Frame(main_container, bg="#f5f5f5")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Build UI
        self.create_header(left_panel)
        self.create_file_upload_section(left_panel)
        self.create_file_list_section(left_panel)
        
        self.create_action_card(right_panel)
        self.create_results_card(right_panel)
        self.create_footer(right_panel)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Primary.TButton',
                       background='#4CAF50',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 9, 'bold'))
        
        style.map('Primary.TButton',
                 background=[('active', '#45a049'), ('disabled', '#cccccc')])
        
        style.configure('Secondary.TButton',
                       background='#2196F3',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 9, 'bold'))
        
        style.map('Secondary.TButton',
                 background=[('active', '#1976D2'), ('disabled', '#cccccc')])
        
        style.configure('Select.TButton',
                       background='#FF9800',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 9, 'bold'))
        
        style.map('Select.TButton',
                 background=[('active', '#F57C00')])
        
        style.configure('Small.TButton',
                       background='#607D8B',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=5,
                       font=('Segoe UI', 8))
        
        style.map('Small.TButton',
                 background=[('active', '#546E7A')])

    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg="#f5f5f5")
        header_frame.pack(fill="x", pady=(0, 15))
        
        icon_label = tk.Label(header_frame, text="🎵", font=("Arial", 32), bg="#f5f5f5")
        icon_label.pack()
        
        title = tk.Label(
            header_frame,
            text="Music Classifier",
            font=("Segoe UI", 18, "bold"),
            bg="#f5f5f5",
            fg="#333333"
        )
        title.pack()

    def create_file_upload_section(self, parent):
        card = tk.Frame(parent, bg="white", relief="flat", bd=1)
        card.pack(fill="x", pady=(0, 15))
        card.configure(highlightbackground="#e0e0e0", highlightthickness=1)
        
        inner = tk.Frame(card, bg="white")
        inner.pack(fill="both", expand=True, padx=15, pady=15)
        
        label = tk.Label(
            inner,
            text="📁 Upload Files",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#333333"
        )
        label.pack(anchor="w", pady=(0, 10))
        
        self.btn_upload = ttk.Button(
            inner,
            text="+ Add Audio Files",
            command=self.load_files,
            style='Select.TButton'
        )
        self.btn_upload.pack(fill="x")
        
        self.lbl_count = tk.Label(
            inner,
            text="0 files loaded",
            font=("Segoe UI", 9),
            bg="white",
            fg="#666666"
        )
        self.lbl_count.pack(anchor="w", pady=(5, 0))

    def create_file_list_section(self, parent):
        card = tk.Frame(parent, bg="white", relief="flat", bd=1)
        card.pack(fill="both", expand=True)
        card.configure(highlightbackground="#e0e0e0", highlightthickness=1)
        
        inner = tk.Frame(card, bg="white")
        inner.pack(fill="both", expand=True, padx=15, pady=15)
        
        header_frame = tk.Frame(inner, bg="white")
        header_frame.pack(fill="x", pady=(0, 10))
        
        label = tk.Label(
            header_frame,
            text="📋 File List",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#333333"
        )
        label.pack(side="left")
        
        btn_clear = ttk.Button(
            header_frame,
            text="Clear All",
            command=self.clear_all_files,
            style='Small.TButton'
        )
        btn_clear.pack(side="right")
        
        # Scrollable frame for file list
        list_frame = tk.Frame(inner, bg="#f9f9f9")
        list_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(list_frame, bg="#f9f9f9", highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        
        self.file_list_frame = tk.Frame(canvas, bg="#f9f9f9")
        
        self.file_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.file_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.canvas = canvas

    def create_action_card(self, parent):
        card = tk.Frame(parent, bg="white", relief="flat", bd=1)
        card.pack(fill="x", pady=(0, 15))
        card.configure(highlightbackground="#e0e0e0", highlightthickness=1)
        
        inner = tk.Frame(card, bg="white")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        label = tk.Label(
            inner,
            text="⚡ Analysis Options",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#333333"
        )
        label.pack(anchor="w", pady=(0, 15))
        
        btn_frame = tk.Frame(inner, bg="white")
        btn_frame.pack(fill="x")
        
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        self.btn_classify_all = ttk.Button(
            btn_frame,
            text="🎯 Classify All",
            command=self.classify_all_files,
            state="disabled",
            style='Primary.TButton'
        )
        self.btn_classify_all.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        self.btn_nmf_selected = ttk.Button(
            btn_frame,
            text="🔊 NMF Selected",
            command=self.process_selected_nmf,
            state="disabled",
            style='Secondary.TButton'
        )
        self.btn_nmf_selected.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def create_results_card(self, parent):
        card = tk.Frame(parent, bg="white", relief="flat", bd=1)
        card.pack(fill="both", expand=True, pady=(0, 15))
        card.configure(highlightbackground="#e0e0e0", highlightthickness=1)
        
        inner = tk.Frame(card, bg="white")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        label = tk.Label(
            inner,
            text="📊 Results",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#333333"
        )
        label.pack(anchor="w", pady=(0, 10))
        
        text_frame = tk.Frame(inner, bg="#f9f9f9", relief="flat")
        text_frame.pack(fill="both", expand=True)
        
        self.txt_result = tk.Text(
            text_frame,
            height=15,
            font=("Consolas", 9),
            bg="#f9f9f9",
            fg="#333333",
            relief="flat",
            padx=15,
            pady=15,
            wrap="word",
            borderwidth=0
        )
        self.txt_result.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.txt_result.yview)
        scrollbar.pack(side="right", fill="y")
        self.txt_result.config(yscrollcommand=scrollbar.set)
        
        self.txt_result.insert("1.0", "Ready to analyze audio files...\n\nPlease upload .wav files to begin.")
        self.txt_result.config(state="disabled")

    def create_footer(self, parent):
        footer = tk.Label(
            parent,
            text="Powered by Librosa & scikit-learn",
            font=("Segoe UI", 9),
            bg="#f5f5f5",
            fg="#999999"
        )
        footer.pack(side="bottom", pady=(10, 0))

    def load_files(self):
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        
        for file_path in files:
            if file_path not in [f['path'] for f in self.audio_files]:
                file_info = {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'result': None,
                    'selected': tk.BooleanVar(value=False)
                }
                self.audio_files.append(file_info)
        
        self.update_file_list()
        self.update_button_states()

    def update_file_list(self):
        # Clear existing widgets
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        # Update count
        self.lbl_count.config(text=f"{len(self.audio_files)} file(s) loaded")
        
        # Create file items
        for idx, file_info in enumerate(self.audio_files):
            self.create_file_item(file_info, idx)
        
        # Update canvas scroll region
        self.file_list_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_file_item(self, file_info, idx):
        item_frame = tk.Frame(self.file_list_frame, bg="white", relief="solid", bd=1)
        item_frame.pack(fill="x", pady=5, padx=5)
        
        # Checkbox and name
        top_frame = tk.Frame(item_frame, bg="white")
        top_frame.pack(fill="x", padx=10, pady=8)
        
        cb = tk.Checkbutton(
            top_frame,
            variable=file_info['selected'],
            bg="white",
            font=("Segoe UI", 9)
        )
        cb.pack(side="left")
        
        name_label = tk.Label(
            top_frame,
            text=file_info['name'],
            font=("Segoe UI", 9),
            bg="white",
            fg="#333333",
            anchor="w"
        )
        name_label.pack(side="left", fill="x", expand=True, padx=5)
        
        # Buttons frame
        btn_frame = tk.Frame(top_frame, bg="white")
        btn_frame.pack(side="right")
        
        # Play button
        play_btn = tk.Button(
            btn_frame,
            text="▶",
            command=lambda: self.play_audio(file_info),
            bg="#4CAF50",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2"
        )
        play_btn.pack(side="left", padx=2)
        
        # Stop button
        stop_btn = tk.Button(
            btn_frame,
            text="⬛",
            command=self.stop_audio,
            bg="#f44336",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2"
        )
        stop_btn.pack(side="left", padx=2)
        
        # Delete button
        del_btn = tk.Button(
            btn_frame,
            text="🗑",
            command=lambda: self.delete_file(idx),
            bg="#9E9E9E",
            fg="white",
            font=("Arial", 8),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2"
        )
        del_btn.pack(side="left", padx=2)
        
        # Result label
        if file_info['result']:
            result_label = tk.Label(
                item_frame,
                text=f"Result: {file_info['result']}",
                font=("Segoe UI", 8),
                bg="white",
                fg="#4CAF50",
                anchor="w"
            )
            result_label.pack(fill="x", padx=10, pady=(0, 8))

    def play_audio(self, file_info):
        try:
            if self.current_playing:
                pygame.mixer.music.stop()
            
            pygame.mixer.music.load(file_info['path'])
            pygame.mixer.music.play()
            self.current_playing = file_info['path']
            
            self.txt_result.config(state="normal")
            self.txt_result.insert("1.0", f"▶ Playing: {file_info['name']}\n")
            self.txt_result.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {str(e)}")

    def stop_audio(self):
        try:
            pygame.mixer.music.stop()
            self.current_playing = None
        except:
            pass

    def delete_file(self, idx):
        del self.audio_files[idx]
        self.update_file_list()
        self.update_button_states()

    def clear_all_files(self):
        if self.audio_files and messagebox.askyesno("Confirm", "Clear all files?"):
            self.stop_audio()
            self.audio_files.clear()
            self.update_file_list()
            self.update_button_states()

    def update_button_states(self):
        if self.audio_files:
            self.btn_classify_all.config(state="normal")
            self.btn_nmf_selected.config(state="normal")
        else:
            self.btn_classify_all.config(state="disabled")
            self.btn_nmf_selected.config(state="disabled")

    def classify_all_files(self):
        if not self.audio_files:
            return
        
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", "🔄 Classifying all files...\n\n")
        self.txt_result.config(state="disabled")
        self.root.update()
        
        def classify_thread():
            results = []
            for file_info in self.audio_files:
                try:
                    y, sr = librosa.load(file_info['path'], sr=SR)
                    feats = extract_all_features(y, sr)
                    result = lazy_classify(feats)
                    file_info['result'] = result.upper()
                    results.append(f"✓ {file_info['name']}: {result.upper()}")
                except Exception as e:
                    results.append(f"✗ {file_info['name']}: Error - {str(e)}")
            
            self.root.after(0, lambda: self.display_classify_results(results))
        
        thread = threading.Thread(target=classify_thread)
        thread.daemon = True
        thread.start()

    def display_classify_results(self, results):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", "═" * 50 + "\n")
        self.txt_result.insert(tk.END, "  CLASSIFICATION RESULTS\n")
        self.txt_result.insert(tk.END, "═" * 50 + "\n\n")
        for result in results:
            self.txt_result.insert(tk.END, result + "\n")
        self.txt_result.insert(tk.END, "\n" + "─" * 50 + "\n")
        self.txt_result.insert(tk.END, f"Total: {len(results)} file(s) processed")
        self.txt_result.config(state="disabled")
        self.update_file_list()

    def process_selected_nmf(self):
        selected_files = [f for f in self.audio_files if f['selected'].get()]
        
        if not selected_files:
            messagebox.showwarning("No Selection", "Please select at least one file for NMF analysis.")
            return
        
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", "🔄 Processing NMF separation...\n\n")
        self.txt_result.config(state="disabled")
        self.root.update()
        
        def nmf_thread():
            results = []
            for file_info in selected_files:
                try:
                    results.append(f"\n📁 {file_info['name']}:")
                    files = separate_nmf(file_info['path'])
                    results.append("  ✓ Separated into 2 components")
                    
                    for idx, f in enumerate(files, 1):
                        y, sr = librosa.load(f, sr=SR)
                        feats = extract_all_features(y, sr)
                        res = lazy_classify(feats)
                        results.append(f"  Component {idx}: {res.upper()}")
                except Exception as e:
                    results.append(f"  ✗ Error: {str(e)}")
            
            self.root.after(0, lambda: self.display_nmf_results(results))
        
        thread = threading.Thread(target=nmf_thread)
        thread.daemon = True
        thread.start()

    def display_nmf_results(self, results):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", "═" * 50 + "\n")
        self.txt_result.insert(tk.END, "  NMF SOURCE SEPARATION\n")
        self.txt_result.insert(tk.END, "═" * 50 + "\n")
        for result in results:
            self.txt_result.insert(tk.END, result + "\n")
        self.txt_result.insert(tk.END, "\n" + "─" * 50 + "\n")
        self.txt_result.insert(tk.END, "Analysis completed!")
        self.txt_result.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernMusicApp(root)
    root.mainloop()