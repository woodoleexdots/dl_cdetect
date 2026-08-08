"""3-slide PPTX: model, model comparison, validation results.

Assets are drawn with matplotlib into results/slides_assets/, then a 16:9
deck is assembled at results/NV_C13_hybrid_3slides.pptx.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "slides_assets"
FIGS = ROOT / "results" / "figs"
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc, fontsize=11, tc="black", sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="0.35", lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (0.055 if sub else 0), text, ha="center",
            va="center", fontsize=fontsize, color=tc, weight="bold", wrap=True)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.085, sub, ha="center", va="center",
                fontsize=fontsize - 2.5, color="0.25")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=22, lw=1.8, color="0.2"))


def asset_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, h = 0.56, 0.30
    box(ax, 0.010, y, 0.150, h, "CPMG 신호\nN=8/16/20", "#dbe9ff",
        sub="(3×700, 상온 실측)")
    box(ax, 0.205, y, 0.165, h, "주기-윈도우\n토큰화", "#dff2df",
        sub="61 윈도우 slice-stack")
    box(ax, 0.415, y, 0.165, h, "Transformer\n(윈도우 간 어텐션)", "#dff2df",
        sub="PeriodFormer, 1.1M")
    box(ax, 0.625, y, 0.155, h, "P(spin) 곡선\n→ 후보 영역", "#fff3cc",
        sub="오탐 0 검출")
    box(ax, 0.825, y, 0.165, h, "영역 제약\nDE + BIC", "#ffe0e0",
        sub="클러스터 내 열거")
    for x1, x2 in [(0.160, 0.205), (0.370, 0.415), (0.580, 0.625), (0.780, 0.825)]:
        arrow(ax, x1, y + h / 2, x2, y + h / 2)
    ax.text(0.5, 0.97, "PF→DE 하이브리드 파이프라인", ha="center", fontsize=15,
            weight="bold")
    ax.text(0.287, 0.38, "물리 사전지식:\n스핀은 주기 TP(A)에서\n수직선을 만든다",
            ha="center", va="top", fontsize=9, color="0.3")
    ax.text(0.497, 0.38, "실제 스핀만 이웃 윈도우에\n일관된 흔적 → 어텐션이 종합",
            ha="center", va="top", fontsize=9, color="0.3")
    ax.text(0.907, 0.38, "같은 영역에 스핀 여러 개 허용\n개수는 BIC가 결정",
            ha="center", va="top", fontsize=9, color="0.3")
    # output
    box(ax, 0.35, 0.06, 0.30, 0.18, "출력: {(A∥, A⊥)} 스핀 목록", "#eadcf8",
        sub="NV1 14개 · NV2 10개 (앵커 7/7 회수)")
    arrow(ax, 0.907, y - 0.02, 0.65, 0.17)
    fig.savefig(ASSETS / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def asset_compare():
    fig, ax = plt.subplots(figsize=(12.4, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cols = [
        ("2021 (Jung et al.)", "#f2d5cc",
         "이미지 → 윈도우별\n독립 MLP 61개\n+ AE 디노이저",
         "윈도우 간 정보 공유 없음\n디노이저가 딥 훼손"),
        ("CNN 뱅크 (개선)", "#fde9c8",
         "CNN + N=8/16/20\n3채널 joint 입력",
         "채널 결합으로 개선\n여전히 독립 판정"),
        ("SpinDETR", "#d9ead3",
         "원신호 → set 예측\n(DETR, 물리 사전지식 X)",
         "(A,B) 직접 출력\n1 forward pass"),
        ("PeriodFormer", "#cfe2f3",
         "윈도우 임베딩 토큰\n+ 교차 윈도우 어텐션",
         "물리 사전지식 유지\n전 조건 오탐 0"),
        ("하이브리드 (최종)", "#ead1dc",
         "PF 후보 영역\n→ 영역 제약 DE",
         "검출 신뢰성 + 열거 능력\n전 조건 1위"),
    ]
    w = 0.184
    for i, (title, fc, body, note) in enumerate(cols):
        x = 0.008 + i * 0.20
        box(ax, x, 0.42, w, 0.44, title, fc, fontsize=11.5)
        ax.text(x + w / 2, 0.52, body, ha="center", va="center", fontsize=9.5)
        ax.text(x + w / 2, 0.24, note, ha="center", va="center", fontsize=8.5,
                color="0.25", style="italic")
    fig.savefig(ASSETS / "compare_schematic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def asset_f1_bars():
    methods = ["2021 풀 파이프라인", "CNN+joint-N", "SpinDETR", "PeriodFormer",
               "cdetect-DE", "하이브리드"]
    f1_room = [0.57, 0.84, 0.88, 0.94, 0.94, None]      # synthetic suite sigma=0.08
    # 50-spin twin arms
    armA = [0.376, None, None, 0.412, 0.714, 0.780]
    armB = [0.367, None, None, 0.357, 0.500, 0.510]
    armC = [0.353, None, None, 0.338, 0.480, 0.553]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    names = methods[:5]
    vals = f1_room[:5]
    colors = ["#b45f4d", "#e8a33d", "#6aa84f", "#3d85c6", "#999999"]
    ax.barh(range(len(names)), vals, color=colors[: len(names)], alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 @ 상온 노이즈 (합성 GT)", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_title("아키텍처 ablation (σ=0.08)", fontsize=11)
    ax.invert_yaxis()

    ax = axes[1]
    x = np.arange(3)
    width = 0.2
    sel = [(0, "2021", "#b45f4d"), (3, "PeriodFormer", "#3d85c6"),
           (4, "cdetect-DE", "#6aa84f"), (5, "하이브리드", "#cc0000")]
    for j, (idx, label, c) in enumerate(sel):
        vals = [armA[idx], armB[idx], armC[idx]]
        ax.bar(x + (j - 1.5) * width, vals, width, label=label, color=c, alpha=0.9)
        for xi, v in zip(x + (j - 1.5) * width, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["암 A: 저온 트윈\n(그들의 조건)", "암 B: 상온\n(우리 조건)",
                        "암 C: 모델 불일치\n(현실 오차)"], fontsize=9.5)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_title("실측 50-스핀 배스 검증 (Delft 공개 데이터, 테스트 전용)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "f1_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_deck():
    from pptx import Presentation
    from pptx.util import Emu, Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def add_title(slide, text, sub=None):
        tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.15),
                                      Inches(12.5), Inches(0.9))
        tf = tb.text_frame
        tf.text = text
        tf.paragraphs[0].font.size = Pt(28)
        tf.paragraphs[0].font.bold = True
        if sub:
            p = tf.add_paragraph()
            p.text = sub
            p.font.size = Pt(13)

    def add_bullets(slide, items, left, top, width, size=13):
        tb = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(2.5))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "• " + it
            p.font.size = Pt(size)

    # ---- slide 1: model ----
    s = prs.slides.add_slide(blank)
    add_title(s, "제안 모델: PF→DE 하이브리드",
              "상온 NV CPMG에서 ¹³C 핵스핀 (A∥, A⊥) 자동 추출")
    s.shapes.add_picture(str(ASSETS / "pipeline.png"), Inches(0.35),
                         Inches(1.15), width=Inches(12.6))
    add_bullets(s, [
        "1단 PeriodFormer: 후보 주기마다 접은 slice-stack을 토큰으로 임베딩 — 물리 사전지식 유지, 윈도우 간 어텐션으로 실제 스핀만 통과 (오탐 0)",
        "2단 영역 제약 DE: PF가 좁힌 영역 안에서만 스핀을 순차 추가(BIC) — 한 클러스터의 다중 스핀까지 열거",
        "학습은 forward model(Eq.1–3) 합성 데이터만 사용 · 실험 데이터와 실측 배스는 테스트 전용",
    ], 0.5, 5.9, 12.3)

    # ---- slide 2: comparison ----
    s = prs.slides.add_slide(blank)
    add_title(s, "아키텍처 비교: 2021 → 하이브리드",
              "같은 조건·같은 합성 GT에서 재학습해 공정 비교 (ablation)")
    s.shapes.add_picture(str(ASSETS / "compare_schematic.png"), Inches(0.35),
                         Inches(1.1), width=Inches(12.6))
    s.shapes.add_picture(str(ASSETS / "f1_bars.png"), Inches(0.35),
                         Inches(4.15), width=Inches(12.6))
    add_bullets(s, [
        "상온 노이즈에서 2021 F1 0.57 → 하이브리드 계열 0.94 · 기여 분해: CNN(+0.14), joint-N(+0.07), 토큰-어텐션(+0.10)",
    ], 0.5, 7.0, 12.3, size=12)

    # ---- slide 3: validation ----
    s = prs.slides.add_slide(blank)
    add_title(s, "검증: 3중 근거",
              "① 합성 GT ② 실측 50-스핀 디지털 트윈(공개 데이터·테스트 전용) ③ 실데이터 교차 수렴")
    s.shapes.add_picture(str(FIGS / "23_method_comparison_NV1.png"),
                         Inches(0.35), Inches(1.15), width=Inches(7.1))
    s.shapes.add_picture(str(FIGS / "21_ensemble_overlay_NV1.png"),
                         Inches(7.65), Inches(1.15), width=Inches(5.35))
    add_bullets(s, [
        "실측 50-스핀 배스(같은 NV 계보, 4TU 공개)에서 하이브리드가 전 조건 1위 — 저온 F1 0.78 / 상온 0.51 / 모델 불일치 0.55 (2021: 0.38/0.37/0.35)",
        "왼쪽: 9개 알고리즘의 NV1 검출이 최종 14스핀(빨간선)에 계단식 수렴 · ppt 앵커 7/7 회수",
        "오른쪽: 최종 14스핀 forward-model이 3개 채널 실데이터를 동시 재현 (RMSE 0.088/0.116/0.176)",
    ], 0.5, 6.15, 12.3, size=12.5)

    out = ROOT / "results" / "NV_C13_hybrid_3slides.pptx"
    prs.save(out)
    print("saved ->", out)


if __name__ == "__main__":
    asset_pipeline()
    asset_compare()
    asset_f1_bars()
    build_deck()
