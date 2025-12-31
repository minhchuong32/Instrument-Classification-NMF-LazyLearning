import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

# --- CẤU HÌNH ---
SR = 22050
CLASSES = ["cel", "cla", "flu", "gac"]  # 4 nhạc cụ ví dụ


# --- BƯỚC 1 & 2: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG ---
def extract_all_features(y, sr):
    # Loại bỏ silent
    y_trim, _ = librosa.effects.trim(y, top_db=25)

    # MFCC (13 coeffs)
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
    cent_mean = np.mean(centroid)
    cent_std = np.std(centroid)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y_trim)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Trả về vector đặc trưng theo đúng thứ tự file CSV của bạn
    features = (
        list(mfcc_mean) + list(mfcc_std) + [cent_mean, cent_std, zcr_mean, zcr_std]
    )
    return features


# --- BƯỚC 3: THUẬT TOÁN PHÂN LOẠI (LAZY LEARNING) ---
def lazy_classify(test_features, train_csv="data\trainsynth_features_detailn.csv"):
    df = pd.read_csv(train_csv)
    # Lấy các cột số (đặc trưng)
    X_train = df.iloc[:, :-2].values
    y_train = df["family"].values

    # Chuẩn hóa (Z-score Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform([test_features])

    # Tính khoảng cách Euclidean (KNN với K=1)
    distances = [distance.euclidean(test_scaled[0], x) for x in X_train_scaled]
    nearest_index = np.argmin(distances)

    return y_train[nearest_index]


# --- TÍNH NĂNG NÂNG CAO: TÁCH NGUỒN NMF ---
def separate_nmf(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    S = np.abs(librosa.stft(y))
    # Phân rã thành 2 thành phần (giả lập 2 nhạc cụ)
    W, H = librosa.decompose.decompose(S, n_components=2, sort=True)

    output_files = []
    for i in range(2):
        S_i = np.outer(W[:, i], H[i, :])
        # Tái tạo âm thanh dùng pha của tín hiệu gốc
        y_out = librosa.istft(S_i * np.exp(1j * np.angle(librosa.stft(y))))
        out_name = f"extracted_part_{i+1}.wav"
        sf.write(out_name, y_out, sr)
        output_files.append(out_name)
    return output_files


# --- GIAO DIỆN CHÍNH (TKINTER) ---
class MusicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XLTN: Nhận diện Nhạc cụ & Tách Nguồn Âm thanh bằng NMF")
        self.root.geometry("500x450")

        tk.Label(
            root, text="ĐỒ ÁN XỬ LÝ TÍN HIỆU ÂM THANH", font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Nút chọn file
        self.btn_select = tk.Button(
            root, text="Chọn file âm thanh (.wav)", command=self.load_file, bg="#e1e1e1"
        )
        self.btn_select.pack(pady=10)

        self.lbl_file = tk.Label(root, text="Chưa chọn file", fg="grey")
        self.lbl_file.pack()

        # Nút chức năng
        self.btn_classify = tk.Button(
            root,
            text="Phân loại đơn (Lazy Learning)",
            command=self.classify,
            state="disabled",
            bg="#90ee90",
        )
        self.btn_classify.pack(pady=5)

        self.btn_nmf = tk.Button(
            root,
            text="Tách nguồn NMF & Nhận diện",
            command=self.process_nmf,
            state="disabled",
            bg="#add8e6",
        )
        self.btn_nmf.pack(pady=5)

        # Kết quả
        self.txt_result = tk.Text(root, height=10, width=50)
        self.txt_result.pack(pady=20)

        self.current_file = None

    def load_file(self):
        self.current_file = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav")]
        )
        if self.current_file:
            self.lbl_file.config(text=os.path.basename(self.current_file), fg="black")
            self.btn_classify.config(state="normal")
            self.btn_nmf.config(state="normal")

    def classify(self):
        try:
            y, sr = librosa.load(self.current_file, sr=SR)
            feats = extract_all_features(y, sr)
            result = lazy_classify(feats)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(
                tk.END, f"--- KẾT QUẢ ---\nNhạc cụ dự đoán: {result.upper()}"
            )
        except Exception as e:
            messagebox.showerror(
                "Lỗi", f"Vui lòng kiểm tra file nsynth_features_detail.csv\n{str(e)}"
            )

    def process_nmf(self):
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "Đang tách nguồn bằng NMF... Vui lòng đợi...\n")
        self.root.update()

        try:
            files = separate_nmf(self.current_file)
            self.txt_result.insert(tk.END, "Đã tách thành công 2 nguồn âm.\n")
            for f in files:
                y, sr = librosa.load(f, sr=SR)
                feats = extract_all_features(y, sr)
                res = lazy_classify(feats)
                self.txt_result.insert(tk.END, f">> Thành phần {f}: {res.upper()}\n")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicApp(root)
    root.mainloop()
