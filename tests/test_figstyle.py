import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from eegoddball import figstyle
    from eegoddball.config import FIGURES_DIR
except ImportError:
    from src.eegoddball import figstyle
    from src.eegoddball.config import FIGURES_DIR


def test_figstyle_apply_and_save():
    figstyle.apply()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=figstyle.COLORS["target"], label="Target")
    ax.set_xlabel(figstyle.LABELS["time"])
    ax.set_ylabel(figstyle.LABELS["amplitude"])

    test_path = figstyle.save(
        fig, "test_style_output", folder=FIGURES_DIR, save_pdf=False, close=True
    )
    assert os.path.exists(test_path)
    if os.path.exists(test_path):
        os.remove(test_path)
