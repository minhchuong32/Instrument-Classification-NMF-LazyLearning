import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

SR = 22050
OUTPUT_DIR = "results"

# Tạo thư mục lưu kết quả nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def separate_instruments_nmf(input_file, output_prefix="separated"):
    print(f"--- Đang bắt đầu tách nguồn NMF cho file: {input_file} ---")

    # Load file âm thanh bị chồng chéo
    y, sr = librosa.load(input_file, sr=SR)

    # Chuyển sang miền tần số (STFT) - Lấy Magnitude (biên độ)
    S = np.abs(librosa.stft(y))

    # Áp dụng NMF (V = W * H)
    # n_components=2: Tách thành 2 nguồn âm thanh
    W, H = librosa.decompose.decompose(S, n_components=2, sort=True)

    # Tái tạo và lưu từng nguồn âm thanh
    separated_files = []
    for i in range(len(W[0])):
        # Phục hồi phổ của từng thành phần: S_i = W_i * H_i
        S_i = np.outer(W[:, i], H[i, :])

        # Kết hợp với Phase (pha) của tín hiệu gốc để chuyển về miền thời gian
        # Dùng np.angle để lấy pha, giúp âm thanh sau tách tự nhiên hơn
        y_out = librosa.istft(S_i * np.exp(1j * np.angle(librosa.stft(y))))

        # Thiết lập đường dẫn lưu file vào thư mục results
        out_name = os.path.join(OUTPUT_DIR, f"{output_prefix}_{i+1}.wav")

        # Lưu file xuống ổ cứng
        sf.write(out_name, y_out, sr)
        separated_files.append(out_name)
        print(f"Đã lưu thành công: {out_name}")

    return separated_files


def create_mixed_testcase(file1, file2, output_file="mixed_test.wav"):
    # Đảm bảo lưu file mixed vào thư mục results cho đồng bộ
    output_path = os.path.join(OUTPUT_DIR, output_file)

    y1, sr = librosa.load(file1, sr=SR)
    y2, _ = librosa.load(file2, sr=SR)

    # Cắt cho 2 file bằng độ dài nhau
    min_len = min(len(y1), len(y2))
    y_mixed = y1[:min_len] + y2[:min_len]

    sf.write(output_path, y_mixed, sr)
    print(f"--- Đã tạo và lưu file hỗn hợp (Testcase): {output_path} ---")
    return output_path


if __name__ == "__main__":
    file_mau_1 = "data/archive/processed_nsynth/train/bass_electronic_013-046-050.wav"
    file_mau_2 = "data/archive/processed_nsynth/train/bass_acoustic_000-061-050.wav"

    if os.path.exists(file_mau_1) and os.path.exists(file_mau_2):
        mixed_file = create_mixed_testcase(file_mau_1, file_mau_2)

        # Tiến hành tách nguồn từ file vừa trộn
        parts = separate_instruments_nmf(mixed_file)

        print("\n" + "=" * 30)
        print("HOÀN THÀNH QUÁ TRÌNH TÁCH NGUỒN")
        print(f"Các file đã được lưu trong thư mục: {os.path.abspath(OUTPUT_DIR)}")
        print("=" * 30)
    else:
        print("!!! LỖI: Không tìm thấy các file mẫu trong đường dẫn 'data/archive/'.")
        print("Vui lòng kiểm tra lại vị trí các file .wav để tạo testcase.")
