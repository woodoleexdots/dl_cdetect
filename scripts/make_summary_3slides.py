"""Ultra-compact 3-slide summary deck.

1. one table: manual (previous ppt) vs 2021-method vs proposed spins
2. validation on the public 50-spin bath (fig 26)
3. NV1/NV2 real CPMG data (N=8/16/20) with fitted overlays (figs 21/22)

Output: results/NV_C13_summary_3slides.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figs"


def add_title(slide, text, sub=None, msg=None):
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.12), Inches(12.5),
                                  Inches(1.0))
    tf = tb.text_frame
    tf.text = text
    tf.paragraphs[0].font.size = Pt(26)
    tf.paragraphs[0].font.bold = True
    if sub:
        p = tf.add_paragraph()
        p.text = sub
        p.font.size = Pt(12)
    if msg:
        p = tf.add_paragraph()
        p.text = "▶ " + msg
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor.from_string("B45000")


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------------- slide 1: three-way spin table ----------------
    s = prs.slides.add_slide(blank)
    add_title(s, "요약 ①: 찾아낸 ¹³C 핵스핀 — 세 방법 비교",
              "값 = (A∥, A⊥) kHz · 제안 방법은 95% 부트스트랩 CI 보유 (예: −87.9±0.9)",
              msg="수동 4개 → 2021 방법 8개(위치만) → 제안 방법 14개(결합강도+오차막대) — 앵커 전부 회수 + 신규 10개")

    headers = ["", "이전 ppt (수동 분석)", "2021 방법 (core-pipeline 재현)",
               "제안 방법 (하이브리드 앙상블)"]
    rows = [
        ("NV1\n검출 스핀",
         "4개\n(−88, 47)\n(−5, 35)\n(+8, 31)\n(−38, 37)*",
         "8개 — A 위치만\n−34, −20, −6, +2,\n+12, +22, +42, +46\n(B는 별도 회귀 필요)",
         "14개 — (A∥, A⊥) 완비\n(−87.9,18) (−18.8,16) (−11.5,14) (−5.1,22)\n(+0.6,17) (+4.0,26) (+8.4,28) (+14.2,22)\n(+24.3,20) (+39.0,28) (+48.0,25) (+65.6,28)\n(+91.9,33) (+114.4,24)"),
        ("NV2\n검출 스핀",
         "3개\n(−340, 290)\n(+150, 110)\n(−42, 150)*",
         "— (미평가:\n단일 채널·강결합은\n원방법 적용 범위 밖)",
         "10개 — 앵커 3/3 회수 포함\n(−151.2,95) (−58.9,182) (−45.9,50)\n(−38.9,67) (−12.6,42) (−2.7,37)\n(+51.7,91) (+55.9,55) (+346.9,261)\n(+430.7,58)⚠"),
        ("신뢰 근거",
         "눈 판독\n(대략 위치 앵커)",
         "문턱값 피크\n오탐 포함 (P 0.55~0.78)",
         "BIC k*=14 = RJMCMC 최빈 15\nCI 상호 비겹침 · 잔차 노이즈 바닥\n실데이터 RMSE 0.088/0.116/0.176"),
    ]
    tblshape = s.shapes.add_table(len(rows) + 1, 4, Inches(0.35), Inches(1.55),
                                  Inches(12.65), Inches(5.6))
    tbl = tblshape.table
    tbl.columns[0].width = Inches(1.15)
    tbl.columns[1].width = Inches(2.6)
    tbl.columns[2].width = Inches(3.3)
    tbl.columns[3].width = Inches(5.6)
    fills = [None, "FCE5CD", "F2D5CC", "D9EAD3"]
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(12)
        c.text_frame.paragraphs[0].font.bold = True
        if fills[j]:
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string(fills[j])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i + 1, j)
            tf = c.text_frame
            for li, line in enumerate(val.split("\n")):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.text = line
                p.font.size = Pt(9.5 if j == 3 else 10)
                if li == 0:
                    p.font.bold = True
    tb = s.shapes.add_textbox(Inches(0.4), Inches(7.12), Inches(12.5), Inches(0.35))
    tb.text_frame.text = ("*수동 분석의 A∥ 부호는 m_s 브랜치 관례 차이 · "
                          "⚠ = 단일 채널 축퇴로 보류 · 2021 방법 수치는 core-pipeline 재현 기준(원방법 하한)")
    tb.text_frame.paragraphs[0].font.size = Pt(10)

    # ---------------- slide 2: 50-spin validation ----------------
    s = prs.slides.add_slide(blank)
    add_title(s, "요약 ②: 공개 실측 50-스핀 배스로 검증",
              "van de Stolpe 2024 (4TU 공개) — Jung 2021과 같은 NV 계보 · 정답을 아는 유일한 실제 스핀 환경",
              msg="저온 트윈 21/27 회수(RMSE 0.074) · 상온 트윈 11/27·오탐 0 — 스핀 값은 학습에 미사용(순수 테스트)")
    s.shapes.add_picture(str(FIGS / "26_twin_validation.png"), Inches(1.2),
                         Inches(1.45), width=Inches(10.6))
    tb = s.shapes.add_textbox(Inches(0.4), Inches(6.95), Inches(12.5), Inches(0.5))
    tf = tb.text_frame
    tf.text = ("• 벤치마크 종합(F1): 제안 하이브리드 v2가 저온 0.84 / 상온 0.57 / 오차주입 0.61로 전 조건 1위 "
               "(2021 재현 0.38/0.37/0.35, 고전 DE 0.71/0.50/0.48, RJMCMC 0.84/0.56/0.52)")
    tf.paragraphs[0].font.size = Pt(12)

    # ---------------- slide 3: real-data fits ----------------
    s = prs.slides.add_slide(blank)
    add_title(s, "요약 ③: 실제 CPMG 데이터 (NV1: N=8/16/20 · NV2: N=16)와 최종 피팅",
              "회색 = 실험 · 빨강 = 검출 스핀들의 forward model (그린 것이 아니라 물리 공식으로 계산된 예측)",
              msg="NV1 14스핀이 세 측정(N=8/16/20)을 동시에 재현(RMSE 0.088/0.116/0.176) · NV2 10스핀 RMSE 0.286")
    s.shapes.add_picture(str(FIGS / "21_ensemble_overlay_NV1.png"),
                         Inches(0.25), Inches(1.55), width=Inches(6.6))
    s.shapes.add_picture(str(FIGS / "22_ensemble_overlay_NV2.png"),
                         Inches(7.0), Inches(1.55), width=Inches(6.1))
    tb = s.shapes.add_textbox(Inches(0.4), Inches(6.85), Inches(12.5), Inches(0.55))
    tf = tb.text_frame
    tf.text = ("• 왼쪽 NV1: 위→아래 N=8/16/20 — 같은 14개 스핀 목록 하나가 세 데이터를 모두 설명    "
               "• 오른쪽 NV2: 강결합 프린지 구조까지 추적 (단일 채널 한계로 1개 스핀 보류)")
    tf.paragraphs[0].font.size = Pt(12)

    out = ROOT / "results" / "NV_C13_summary_3slides.pptx"
    prs.save(out)
    print("saved ->", out)


if __name__ == "__main__":
    main()
