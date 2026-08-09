"""Assets for the five audience-question slides (NV experimentalist level).

q1_bic.png     : how the spin COUNT is determined — NV1 BIC trace (min at
                 k*=14) + RJMCMC posterior P(k) on the same real data
q2_repro.png   : what exactly was reproduced from the 2021 pipeline
q3_lineage.png : provenance/trust chain of the public 50-spin dataset
q5_ci.png      : why the NV1 list is credible — bootstrap CIs + anchors
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "slides_assets"
BV2 = ROOT / "results" / "benchmark_v2"

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# NV1 ensemble BIC trace (from ensemble_run.log)
KS = list(range(1, 18))
BIC = [-6603.4, -6964.4, -7255.1, -7478.9, -7782.1, -7884.6, -7995.1,
       -8099.8, -8157.7, -8199.8, -8245.0, -8250.6, -8252.0, -8285.1,
       -8275.0, -8264.0, -8251.6]
RSS = [89.83, 75.09, 64.91, 57.93, 49.78, 47.06, 44.32, 41.86, 40.43,
       39.34, 38.22, 37.84, 37.54, 36.69, 36.60, 36.52, 36.47]


def q1_bic():
    # RJMCMC posterior P(k) on real NV1 (region-constrained)
    from cpmg.represent import envelope_normalize
    from cpmg.rjmcmc import RJMCMC
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau = nv1["a"].to_numpy(float)
    m = np.array([envelope_normalize(tau, nv1[c].to_numpy(float))[0]
                  for c in ["CPMG8", "CPMG16", "CPMG20"]])
    regions = [tuple(np.array(r) * 1e3) for r in json.loads(
        (BV2 / "hybrid_results.json").read_text())["nv1_ensemble"]["regions_khz"]]
    mc = RJMCMC(tau, m, (8, 16, 20), 440.1, sigma=0.11, max_spins=22,
                seed=0, regions=regions)
    out = mc.run(n_iter=40000)
    pk = np.array(out["posterior_k"])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.9))
    ax = axes[0]
    ax.plot(KS, BIC, "o-", color="tab:red", label="BIC (적합도 − 벌점)")
    ax.axvline(14, color="tab:red", ls="--", lw=1)
    ax.annotate("k*=14 (최소)\n15번째 스핀부터는\n'월세'를 못 냄", (14, BIC[13]),
                xytext=(15.2, -8100), fontsize=10, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax2 = ax.twinx()
    ax2.plot(KS, RSS, "s-", color="0.5", alpha=0.7, label="잔차 (RSS)")
    ax2.set_ylabel("잔차 RSS", color="0.4")
    ax.set_xlabel("가정한 스핀 수 k")
    ax.set_ylabel("BIC", color="tab:red")
    ax.set_title("① NV1 실데이터: 스핀을 하나씩 추가하며 BIC 추적", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ks = np.arange(len(pk))
    sel = pk > 0.001
    ax.bar(ks[sel], pk[sel], color="tab:blue", alpha=0.85)
    ax.set_xlabel("스핀 수 k")
    ax.set_ylabel("P(k | 데이터)")
    ax.set_title("② 독립 교차확인: RJMCMC가 계산한 스핀 수의 확률분포 (NV1)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "q1_bic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("q1 done; RJMCMC modal k =", out["modal_k"], flush=True)


def q2_repro():
    fig, ax = plt.subplots(figsize=(12.6, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    done = ["slice-stack 주기 접기 표현 (Fig.2b)", "Dense 2048–1024–512 MLP (코드 동일 구조)",
            "3-class 분류 (스핀 0/1/2개)", "1D conv-AE 디노이저 (kernel 4, 64ch)",
            "윈도우별 독립 모델 뱅크 + 피크 검출"]
    not_done = ["N=256 병행 분석 (우리 데이터엔 N≤20뿐)", "계층(hierarchical) 모델",
                "하이퍼파라미터 앙상블", "PSO 파인튜닝"]
    ax.text(0.25, 0.95, "재현한 것 (core pipeline)", fontsize=13, weight="bold",
            ha="center", color="#1a7a1a")
    for i, t in enumerate(done):
        ax.text(0.03, 0.82 - i * 0.14, "✓ " + t, fontsize=11, color="#1a7a1a")
    ax.text(0.75, 0.95, "재현하지 않은 것 (명시적 한정)", fontsize=13,
            weight="bold", ha="center", color="#aa3333")
    for i, t in enumerate(not_done):
        ax.text(0.55, 0.82 - i * 0.14, "✗ " + t, fontsize=11, color="#aa3333")
    ax.text(0.5, 0.10, "충실성 검증: 재현 파이프라인이 '작동해야 할 조건'(저온 트윈, N=32)에서 "
            "정상 작동(오탐 0, val acc 0.73)을 확인 — 실패는 상온 조건에서만 발생",
            fontsize=11, ha="center", weight="bold", color="#1155CC")
    ax.text(0.5, 0.02, "따라서 논문에서는 '2021 core-pipeline reproduction'으로 표기하고, "
            "보고 수치는 원방법 성능의 하한임을 명시", fontsize=10, ha="center",
            color="0.35")
    fig.savefig(ASSETS / "q2_repro.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("q2 done", flush=True)


def q3_lineage():
    fig, ax = plt.subplots(figsize=(12.6, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                    fc=fc, ec="0.35", lw=1.2))
        ax.text(x + w/2, y + h - 0.10, title, ha="center", fontsize=10.5,
                weight="bold")
        ax.text(x + w/2, y + 0.16, sub, ha="center", va="center", fontsize=8.5,
                color="0.3")

    def arr(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.5, color="0.2"))

    y, h = 0.42, 0.42
    box(0.01, y, 0.21, h, "같은 Delft NV 샘플",
        "Abobeih 2019 Nature\n27-스핀 3D 이미징", "#dbe9f8")
    arr(0.225, 0.245, y + h/2)
    box(0.25, y, 0.21, h, "Jung 2021 npj QI",
        "우리가 재현·비교한 논문\n(같은 NV, B=403.553 G)", "#f2d5cc")
    arr(0.465, 0.485, y + h/2)
    box(0.49, y, 0.24, h, "van de Stolpe 2024\nNat. Commun.",
        "50-스핀 네트워크 매핑\nSEDOR 이중공명, 1.8 Hz 분해능", "#dff2df")
    arr(0.735, 0.755, y + h/2)
    box(0.76, y, 0.23, h, "4TU 공개 데이터\n(DOI 10.4121/aba1cc84…)",
        "50개 (A∥, A⊥) + 위치\n+ 핵-핵 결합 실측", "#eadcf8")
    ax.text(0.5, 0.24, "신뢰 근거 3가지", fontsize=12, weight="bold", ha="center")
    ax.text(0.5, 0.11, "① 정답을 잰 기법(SEDOR 이중공명)이 우리 방법(CPMG)과 완전히 독립 — 순환 논증 없음    "
            "② 같은 NV 계보·같은 자기장(403.553 G 소수점까지 일치)    "
            "③ Nature/npj/Nat.Commun. 3중 피어리뷰를 거친 값",
            fontsize=10, ha="center", color="0.25")
    fig.savefig(ASSETS / "q3_lineage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("q3 done", flush=True)


def q5_ci():
    boot = json.loads((BV2 / "nv1_bootstrap.json").read_text())["spins"]
    fig, ax = plt.subplots(figsize=(12.6, 3.9))
    for j, s in enumerate(boot):
        a, ci = s["A_khz"], s["A_ci"]
        ax.errorbar(a, 1.0, xerr=[[a - ci[0]], [ci[1] - a]], fmt="o", ms=6,
                    color="tab:blue", capsize=4, lw=2)
        ax.annotate(f"{a:+.0f}", (a, 1.0), xytext=(a, 1.25 if j % 2 == 0 else 0.72),
                    fontsize=8.5, ha="center", color="tab:blue")
    for a in [-88, 8, -38, -5]:
        ax.plot([a, -a] if a == -38 else [a], [0.4] * (2 if a == -38 else 1),
                "D", ms=8, color="tab:orange", alpha=0.8)
    ax.plot([], [], "D", color="tab:orange", label="수동 분석 앵커 (참고용)")
    ax.errorbar([], [], fmt="o", color="tab:blue", label="최종 14스핀 (95% 부트스트랩 CI)")
    ax.set_xlim(-125, 125)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([])
    ax.set_xlabel("A∥ (kHz)")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("NV1 최종 14스핀: 오차막대(95% CI)가 서로 겹치지 않음 = 통계적으로 구별되는 실체",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "q5_ci.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("q5 done", flush=True)


if __name__ == "__main__":
    q2_repro()
    q3_lineage()
    q5_ci()
    q1_bic()
