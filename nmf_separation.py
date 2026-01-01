import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- CẤU HÌNH HỆ THỐNG ---
SR = 22050  # Tốc độ lấy mẫu tiêu chuẩn
OUTPUT_DIR = "results"  # Thư mục lưu kết quả

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def plot_results(original_mix, separated_sources, sr):
    """
    Hàm vẽ đồ thị dạng sóng (Waveform) để đưa vào báo cáo mục 4.4
    """
    plt.figure(figsize=(10, 8))

    # Vẽ tín hiệu hỗn hợp ban đầu
    plt.subplot(3, 1, 1)  # Chia làm 3 hàng, 1 cột, vị trí 1
    librosa.display.waveshow(original_mix, sr=sr, color="gray", alpha=0.8)
    plt.title("Hỗn hợp gốc (Mixed Signal) - Trước khi tách")
    plt.xlabel("")  # Ẩn nhãn trục x cho gọn

    # Vẽ nguồn tách thứ nhất
    plt.subplot(3, 1, 2)  # Vị trí 2
    librosa.display.waveshow(separated_sources[0], sr=sr, color="blue")
    plt.title("Nguồn đã tách 1 (Component 1)")
    plt.xlabel("")

    # Vẽ nguồn tách thứ hai
    plt.subplot(3, 1, 3)  # Vị trí 3
    librosa.display.waveshow(separated_sources[1], sr=sr, color="green")
    plt.title("Nguồn đã tách 2 (Component 2)")

    plt.tight_layout()  # Tự động căn chỉnh các biểu đồ không bị đè lên nhau

    # Lưu hình ảnh phục vụ minh họa báo cáo
    image_path = os.path.join(OUTPUT_DIR, "minh_hoa_tach_am.png")
    plt.savefig(image_path)
    print(f"--- Đã lưu ảnh minh họa tại: {image_path} ---")
    plt.show()


def separate_instruments_nmf(input_file):
    """
    Thuật toán NMF: Tách file âm thanh hỗn hợp thành 2 nguồn độc lập
    """
    print(f"--- Đang xử lý tách nguồn: {input_file} ---")

    # Tải file âm thanh
    y, sr = librosa.load(input_file, sr=SR)

    # Chuyển sang miền tần số (STFT)
    # stft_complex giữ cả biên độ (độ lớn) và pha (thông tin thời gian)
    stft_complex = librosa.stft(y)
    S = np.abs(stft_complex)  # Ma trận biên độ (đầu vào cho NMF)
    phase = np.exp(1j * np.angle(stft_complex))  # Ma trận pha (để khôi phục âm thanh)

    # Thực hiện phân rã NMF (V ≈ W * H)
    # n_components=2 vì ta muốn tách thành 2 loại nhạc cụ
    print("Đang chạy thuật toán NMF...")
    W, H = librosa.decompose.decompose(S, n_components=2, sort=True)

    separated_audios = []

    # Tái tạo từng nguồn âm thanh
    for i in range(W.shape[1]):
        # Tính toán phổ riêng lẻ của nguồn i: S_i = W_i * H_i
        S_i = np.outer(W[:, i], H[i, :])

        # Kết hợp biên độ đã tách (S_i) với pha gốc để âm thanh tự nhiên
        y_out = librosa.istft(S_i * phase)

        # Lưu file kết quả
        out_name = os.path.join(OUTPUT_DIR, f"tach_nguon_{i+1}.wav")
        sf.write(out_name, y_out, sr)
        separated_audios.append(y_out)
        print(f"Thành công: {out_name}")

    # Gọi hàm vẽ đồ thị minh họa kết quả
    plot_results(y, separated_audios, sr)

    return separated_audios


def create_mixed_testcase(file1, file2):
    """
    Hàm tạo dữ liệu giả lập: Trộn 2 file nhạc cụ khác nhau thành 1
    """
    print("--- Đang tạo file hỗn hợp để thử nghiệm ---")
    y1, _ = librosa.load(file1, sr=SR)
    y2, _ = librosa.load(file2, sr=SR)

    # Cắt hai đoạn âm thanh cho bằng độ dài nhau
    length = min(len(y1), len(y2))
    y_mixed = y1[:length] + y2[:length]

    # Chuẩn hóa biên độ (tránh âm thanh bị rè khi vượt ngưỡng 1.0)
    y_mixed = y_mixed / np.max(np.abs(y_mixed))

    mixed_path = os.path.join(OUTPUT_DIR, "mixed_test_input.wav")
    sf.write(mixed_path, y_mixed, SR)
    return mixed_path


if __name__ == "__main__":
    path1 = "data/archive/processed_nsynth/train/bass_electronic_013-046-050.wav"
    path2 = "data/archive/processed_nsynth/train/bass_acoustic_000-061-050.wav"

    if os.path.exists(path1) and os.path.exists(path2):
        # Tạo file mix
        mixed_file = create_mixed_testcase(path1, path2)

        # Tách file mix bằng NMF
        separate_instruments_nmf(mixed_file)

        print("\nQUÁ TRÌNH HOÀN TẤT. KIỂM TRA THƯ MỤC 'results'")
    else:
        print("LỖI: Không tìm thấy file đầu vào. Vui lòng kiểm tra lại đường dẫn.")
