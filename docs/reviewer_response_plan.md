# 리뷰어 공통 지적 대응 계획 (2026-08-09)

## 항목 1 — 다중 시드 + 오차막대 + 유의성 [실행 중]

- `scripts/multiseed_study.py`: pf_room 학습 시드 5개(암 B/C), pf_cryo 시드 3개(암 A),
  DE 옵티마이저 시드 3개. 산출: 방법×암별 F1 mean±std + 실현(realization) 단위로 짝지은
  Wilcoxon signed-rank p값 (하이브리드 vs DE).
- 결과 파일: `results/benchmark_v2/multiseed.json` (`summary` 키).
- 논문 반영: 모든 벤치마크 표의 수치를 mean±std로 교체, 본문에 p값 명기.

## 항목 2 — RJMCMC 베이지안 베이스라인 [실행 중]

- `cpmg/rjmcmc.py`: birth/death/perturb + 어닐링 + BIC형 복잡도 prior의 네이티브
  trans-dimensional MCMC (old-cdetect §18.5 재구현, 동일 forward model 공유로 공정 비교).
- `scripts/rjmcmc_benchmark.py`: 3개 암 동일 프로토콜 평가 + posterior P(k) 기록.
- 합성 검증: 3-스핀 GT 완전 복원(2초/15k iter) 확인 완료.
- 논문 반영: 비교표에 RJMCMC 열 추가. 중간 결과(암 B: F1 0.556, P 0.944, map k≈11)는
  "무제한 k 탐색의 recall 이득 vs 하이브리드의 precision(1.0) 우위"로 서술 —
  결과가 어느 쪽이든 정직하게 기재하고, RJMCMC의 장점(posterior 제공)은 항목 4와 연결.

## 항목 3 — "2021 재현" 한정 명시 + full protocol 대비 [대응안]

### 3a. 표기 수정 (즉시 적용 가능)
- 모든 표·그림·본문에서 "2021" → **"2021 core-pipeline reproduction"**으로 통일하고
  방법 절에 한정 문구 삽입:
  > "Our baseline reproduces the core HPC classification pipeline of Jung et al.
  > (single-N window-bank MLP + conv-AE denoiser). It intentionally omits the
  > N=256 co-analysis, hierarchical models, hyperparameter ensembling, power-law
  > preprocessing, and PSO fine-tuning of the full protocol; reported baseline
  > numbers therefore lower-bound the original method."
- F1 0.38/0.37/0.35는 "원방법의 성능"이 아니라 "핵심 파이프라인의 우리-조건 성능"임을
  명시 — 저자 반박(W1)의 예봉을 선제 차단.

### 3b. full-protocol 보강 실험 (요청 시 1-2일)
1. **N=256 트윈 암 추가**: 실측 50-스핀 배스를 N=32+N=256(7000pt) 이중 채널로 합성,
   2021 파이프라인을 두 N에 각각 적용 후 교차 확인(원논문 절차) — `benchmark_v2.py`의
   ARMS에 `A2_cryo_full` 추가로 구현 가능.
2. **앙상블·전처리 복원**: (mini_batch, lr, optimizer) 3조합 앙상블 + B<12 kHz 윈도우
   거듭제곱 전처리(1−M^8)를 `ablation.py` 옵션으로 추가.
3. 계층(hierarchical) 모델은 원 코드 사양이 불완전하므로 "구현 범위 밖"으로 명시하고
   저자 데이터/코드 협조 요청 문안 준비(아래).
4. **저자 협업 제안 문구**: "We invite the original authors to evaluate their full
   pipeline on our published digital-twin benchmark; all data and harness code are
   public." — 데이터 요청 이메일 초안은 khoony.jung@gmail.com / t.h.taminiau@tudelft.nl.

## 항목 4 — 불확실성 정량화 [실행 중 + 확장안]

### 4a. 잔차 부트스트랩 (실행 중)
- `scripts/bootstrap_uncertainty.py`: NV1 최종 14스핀의 (A∥, A⊥) 95% CI —
  잔차 재표집 60회 × 국소 재피팅. 잔차 lag-1 자기상관을 보고해 iid 재표집의
  타당성 명시(상관 크면 블록 부트스트랩으로 교체).
- 논문 반영: 최종 스핀 표의 모든 값에 ±(CI) 병기.

### 4b. RJMCMC posterior와의 정합 (항목 2와 연결)
- RJMCMC의 P(k)·파라미터 사후분포를 부트스트랩 CI와 대조 — 두 독립 방식의 CI가
  겹치면 불확실성 추정의 신뢰도 근거가 됨.

### 4c. (선택) PF 확률 보정
- 합성 스위트에서 reliability curve 산출, 필요시 temperature scaling —
  ICML W3 대응. 구현 30분 거리로 문서화만 해둠.

## 남은 항목 메모
- npj W1(A⊥ 독립 검증)·W2(NV2 추가 N 측정)는 **실험 필요** — 실험팀 논의 안건.
- 베뉴 전략: npj QI/PRApplied 본지 + NeurIPS ML4PS 워크숍 병행.
