# Related Work — 논문 인용용 정리 (2026-08-09)

논문 Related Work 절에 그대로 옮길 수 있도록 (1) 영문 초안, (2) BibTeX, (3) 본문 인용 위치 맵으로 구성.
계열은 세 갈래: 물리 프로토콜 → 딥러닝 → 베이지안/trans-dimensional.

---

## 1. Related Work 영문 초안 (LaTeX-ready)

```latex
\section{Related Work}

\paragraph{Physics-protocol spin detection.}
Individual $^{13}$C nuclear spins around an NV center were first resolved by
dynamical-decoupling spectroscopy, in which each spin imprints periodic
coherence dips described by an analytic product formula~\cite{taminiau2012}.
Building on this forward model, atomic-scale imaging of a 27-spin cluster was
achieved with multidimensional spectroscopy~\cite{abobeih2019}, and correlated
sensing (SEDOR) later mapped a 50-spin network with sub-Hz resolution around
the same NV lineage~\cite{vandestolpe2024}; the associated hyperfine parameters
are public~\cite{delft4tu}. These protocols attain the highest resolution to
date, but require cryogenic operation, long acquisition campaigns, and
extensive manual analysis. A non-learning decomposition algorithm~\cite{oh2020}
automated part of this analysis, identifying 14 spins from $\sim$10,000-point
coherence traces. Our target regime differs: room-temperature CPMG traces of
only 700 points per pulse number, where overlapping dips and strong noise make
manual or purely spectroscopic assignment unreliable.

\paragraph{Deep-learning approaches.}
Jung \emph{et al.}~\cite{jung2021} introduced the first deep-learning pipeline
for this inverse problem: coherence traces are folded at candidate periods into
2D representations and classified by per-window MLP banks, identifying 23
spins from cryogenic $N{=}32/256$ data ($\sim$11{,}000 points). Varona-Uriarte
\emph{et al.}~\cite{varona2024} proposed a signal-to-image network trained on
3.6M synthetic samples for arbitrary magnetic fields. Both output spin
\emph{locations} (or per-window class probabilities) rather than a completed,
uncertainty-quantified spin list, and both were designed for high-SNR
regimes. We reproduce the core pipeline of~\cite{jung2021} under controlled
conditions and show that, while its period-folding representation remains
informative at room temperature (it localizes 8--9 of our 14 confirmed NV1
spins at region level), its isolated per-window decisions degrade into
contiguous high-probability intervals that cannot be decomposed into
individual spins.

\paragraph{Bayesian and trans-dimensional inference.}
Because the number of bath spins is itself unknown, spin detection is a
trans-dimensional model-selection problem, for which reversible-jump
MCMC~\cite{green1995} and BIC-type penalties~\cite{schwarz1978} are the
canonical tools. Concurrently with our work, Poteshman \emph{et
al.}~\cite{poteshman2026} (with the Taminiau group) applied a hybridized
RJMCMC/parallel-tempering sampler to the same NV--$^{13}$C problem,
recovering 45 of 48 reference spins from only 250 cryogenic data points by
constraining the search to DFT-computed hyperfine values at diamond lattice
sites~\cite{takacs2024}; a companion study extends the framework to
high-throughput characterization of spin defects~\cite{poteshman2025}. This
independent adoption of trans-dimensional inference corroborates our design
choice. Our method differs in three respects: (i) the search is narrowed not
by \emph{ab initio} priors---which propagate DFT errors and caused a strong
spin to be missed in~\cite{poteshman2026}---but by a learned attention
detector that proposes candidate regions directly from data; (ii) inference
runs jointly over multiple pulse numbers ($N{=}8/16/20$) at room temperature,
the regime where~\cite{poteshman2026} was not validated and where weak
couplings ($<$25~kHz) were reported unresolved; and (iii) the output is a
completed spin list with bootstrap confidence intervals, with the model
dimension cross-checked by two independent criteria (BIC minimum and RJMCMC
posterior mode).
```

---

## 2. BibTeX (`refs.bib`에 추가)

```bibtex
@article{taminiau2012,
  author  = {Taminiau, T. H. and Wagenaar, J. J. T. and van der Sar, T. and
             Jelezko, F. and Dobrovitski, V. V. and Hanson, R.},
  title   = {Detection and Control of Individual Nuclear Spins Using a
             Weakly Coupled Electron Spin},
  journal = {Physical Review Letters},
  volume  = {109},
  pages   = {137602},
  year    = {2012},
  doi     = {10.1103/PhysRevLett.109.137602}
}

@article{abobeih2019,
  author  = {Abobeih, M. H. and Randall, J. and Bradley, C. E. and
             Bartling, H. P. and Bakker, M. A. and Degen, M. J. and
             Markham, M. and Twitchen, D. J. and Taminiau, T. H.},
  title   = {Atomic-scale imaging of a 27-nuclear-spin cluster using a
             quantum sensor},
  journal = {Nature},
  volume  = {576},
  pages   = {411--415},
  year    = {2019},
  doi     = {10.1038/s41586-019-1834-7}
}

@article{vandestolpe2024,
  author  = {van de Stolpe, G. L. and Kwiatkowski, D. P. and Bradley, C. E.
             and Randall, J. and Abobeih, M. H. and Breitweiser, S. A. and
             Bassett, L. C. and Markham, M. and Twitchen, D. J. and
             Taminiau, T. H.},
  title   = {Mapping a 50-spin-qubit network through correlated sensing},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {2006},
  year    = {2024},
  doi     = {10.1038/s41467-024-46075-4}
}

@misc{delft4tu,
  author    = {van de Stolpe, G. L. and Kwiatkowski, D. P. and
               Bradley, C. E. and Randall, J. and Abobeih, M. H. and
               Breitweiser, S. A. and Bassett, L. C. and Markham, M. and
               Twitchen, D. J. and Taminiau, T. H.},
  title     = {Data underlying the publication: Mapping a 50-spin-qubit
               network through correlated sensing},
  publisher = {4TU.ResearchData},
  year      = {2024},
  doi       = {10.4121/aba1cc84-0aea-4cdc-93ca-68b0db38bd81.v1}
}

@article{oh2020,
  author  = {Oh, Hyunseok and Yun, Jiwon and Abobeih, M. H. and
             Jung, Kyung-Hoon and Kim, Kiho and Taminiau, T. H. and
             Kim, Dohun},
  title   = {Algorithmic decomposition for efficient multiple nuclear spin
             detection in diamond},
  journal = {Scientific Reports},
  volume  = {10},
  pages   = {14884},
  year    = {2020},
  doi     = {10.1038/s41598-020-71339-6}
}

@article{jung2021,
  author  = {Jung, Kyunghoon and Abobeih, M. H. and Yun, Jiwon and
             Kim, Gyeonghun and Oh, Hyunseok and Henry, Ang and
             Taminiau, T. H. and Kim, Dohun},
  title   = {Deep learning enhanced individual nuclear-spin detection},
  journal = {npj Quantum Information},
  volume  = {7},
  pages   = {41},
  year    = {2021},
  doi     = {10.1038/s41534-021-00377-3}
}

@article{varona2024,
  author  = {Varona-Uriarte, B. and Munuera-Javaloy, C. and Terradillos, E.
             and Ban, Y. and Alvarez-Gila, A. and Garrote, E. and
             Casanova, J.},
  title   = {Automatic Detection of Nuclear Spins at Arbitrary Magnetic
             Fields via Signal-to-Image AI Model},
  journal = {Physical Review Letters},
  volume  = {132},
  pages   = {150801},
  year    = {2024},
  doi     = {10.1103/PhysRevLett.132.150801}
}

@article{green1995,
  author  = {Green, Peter J.},
  title   = {Reversible jump Markov chain Monte Carlo computation and
             Bayesian model determination},
  journal = {Biometrika},
  volume  = {82},
  number  = {4},
  pages   = {711--732},
  year    = {1995},
  doi     = {10.1093/biomet/82.4.711}
}

@article{schwarz1978,
  author  = {Schwarz, Gideon},
  title   = {Estimating the Dimension of a Model},
  journal = {The Annals of Statistics},
  volume  = {6},
  number  = {2},
  pages   = {461--464},
  year    = {1978},
  doi     = {10.1214/aos/1176344136}
}

@article{poteshman2026,
  author  = {Poteshman, Abigail N. and Yun, Jiwon and Taminiau, Tim H. and
             Galli, Giulia},
  title   = {Trans-dimensional Hamiltonian model selection and parameter
             estimation from sparse, noisy data},
  journal = {Quantum},
  volume  = {10},
  pages   = {2055},
  year    = {2026},
  doi     = {10.22331/q-2026-04-08-2055},
  note    = {arXiv:2506.18802}
}

@article{poteshman2025,
  author  = {Poteshman, Abigail N. and Onizhuk, Mykyta and
             Egerstrom, Christopher and Mark, Daniel P. and
             Awschalom, David D. and Heremans, F. Joseph and Galli, Giulia},
  title   = {High-throughput spin-bath characterization of spin defects in
             semiconductors},
  journal = {Physical Review Applied},
  volume  = {24},
  pages   = {054048},
  year    = {2025},
  doi     = {10.1103/p57x-8kk7},
  note    = {arXiv:2506.19259}
}

@article{takacs2024,
  author  = {Tak{\'a}cs, Istv{\'a}n and Iv{\'a}dy, Viktor},
  title   = {Accurate hyperfine tensors for solid state quantum
             applications: case of the NV center in diamond},
  journal = {Communications Physics},
  volume  = {7},
  pages   = {178},
  year    = {2024},
  doi     = {10.1038/s42005-024-01668-9}
}
```

---

## 3. 본문 인용 위치 맵

| 본문 위치 | 인용 | 용도 |
|---|---|---|
| Intro 첫 문단 (배경) | taminiau2012, abobeih2019, vandestolpe2024 | DD 분광의 성취와 저온·장시간·수작업 한계 |
| Intro "기존 자동화" | jung2021, varona2024, oh2020 | DL 2종 + 비학습 분해 알고리즘 — 모두 고SNR 전제 |
| Forward model (Eq.1–3) | taminiau2012, jung2021 | 곱 공식의 출처 |
| 방법: RJMCMC 절 | green1995, schwarz1978 | 알고리즘·벌점의 원전 |
| 방법 또는 Related Work | **poteshman2026**, takacs2024 | 동시대 trans-dimensional 연구; DFT-격자 prior와의 대비 (우리: 학습된 영역 제안) |
| Discussion (scale-up) | poteshman2025 | high-throughput 특성화 흐름 속 우리 파이프라인의 위치 |
| 검증 데이터 절 | vandestolpe2024, delft4tu | 50-스핀 GT의 출처와 공개 데이터 DOI |
| 2021 재현 절 | jung2021, oh2020 | core-pipeline 정의·재현 범위 한정 |

## 4. poteshman2026 대비 요지 (리뷰어 대응용 3줄)

1. **탐색 제약의 출처**: 그들은 DFT 격자값(오차가 전파되어 강결합 C36 미검출) — 우리는 데이터에서 학습된 어텐션 영역(사전계산 불필요).
2. **검증 영역**: 그들은 저온 3.7 K·N=32 단일 채널·τ 6–8 µs에서만 검증, 약결합 <25 kHz는 미해결로 명시 — 우리는 상온·3채널 동시·kHz 약결합이 설계 목표이며 실데이터+트윈 3조건에서 검증.
3. **출력 형태**: 그들은 사후 검출률 분포(리스트 미완결) — 우리는 BIC·RJMCMC 이중 확인된 개수와 부트스트랩 CI가 붙은 확정 목록.
