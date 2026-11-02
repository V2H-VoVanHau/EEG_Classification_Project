"""Quy chuẩn đồ họa (Visualization Style) dùng chung cho dự án EEG Oddball.

Sở hữu: TV5.
Tất cả các hình vẽ trong pipeline và báo cáo (Figure 1 -> Figure 10)
đều phải áp dụng `figstyle.apply()` và lưu qua `figstyle.save()`.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from eegoddball.config import FIGURES_DIR
except ImportError:
    from src.eegoddball.config import FIGURES_DIR

# 1. BẢNG MÀU CỐ ĐỊNH (COLORBLIND-FRIENDLY)
COLORS = {
    "standard": "#7f7f7f",     # Xám trung tính cho kích thích thường gặp (S 5)
    "target": "#d62728",       # Đỏ nổi bật cho kích thích đích P300 (S 6)
    "distractor": "#1f77b4",   # Xanh dương cho kích thích gây nhiễu/lạ (S 7)
    "difference": "#9467bd",   # Tím cho sóng hiệu (target - standard)
    "auc": "#2ca02c",          # Xanh lá cho đường cong ROC AUC
    "baseline": "#000000",     # Đen cho đường tham chiếu 0
}

LINESTYLES = {
    "standard": "-",
    "target": "-",
    "distractor": "--",
    "difference": ":",
}

# 2. HẰNG SỐ NHÃN TRỤC & ĐƠN VỊ TIÊU CHUẨN
LABELS = {
    "time": "Time (s)",
    "amplitude": "Amplitude (µV)",
    "auc": "ROC AUC",
    "frequency": "Frequency (Hz)",
    "channels": "Channels",
    "attribution": "Integrated Gradients Attribution",
}


def apply():
    """Thiết lập cấu hình matplotlib.rcParams chuẩn xuất bản khoa học."""
    plt.rcParams.update({
        # Font chữ sạch, sans-serif
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",

        # Đường trục và khung
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",

        # Lưới mờ hỗ trợ đọc số liệu
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,

        # Độ phân giải xuất bản
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    sns.set_palette(list(COLORS.values()))


def save(fig, name, folder=None, save_pdf=True, close=False):
    """Lưu hình chuẩn 300 dpi (.png) và định dạng vector (.pdf).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Đối tượng hình cần lưu.
    name : str
        Tên file (vd: 'fig1_pipeline' hoặc 'fig9_timeresolved.png').
    folder : str, optional
        Thư mục lưu hình, mặc định lưu vào `figures/`.
    save_pdf : bool, default True
        Tự động xuất thêm phiên bản vector PDF phục vụ nhúng Overleaf/LaTeX.
    close : bool, default False
        Đóng figure sau khi lưu để tiết kiệm RAM.
    """
    save_dir = folder if folder else FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Tách phần mở rộng nếu có
    base_name = os.path.splitext(name)[0]

    # Lưu PNG 300 DPI
    png_path = os.path.join(save_dir, f"{base_name}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # Lưu vector PDF
    pdf_path = None
    if save_pdf:
        pdf_path = os.path.join(save_dir, f"{base_name}.pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"[figstyle] Đã lưu hình: {png_path}" + (f" & {pdf_path}" if pdf_path else ""))

    if close:
        plt.close(fig)

    return png_path


# Alias tương thích ngược
apply_style = apply
save_figure = save
