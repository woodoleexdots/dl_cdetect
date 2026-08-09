"""Generate one architecture figure per model (4 models) via PaperBanana
with the local Qwen-Image backend. Minimal-label constraint included in
every caption (validated to give clean text)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import create_sample_inputs, get_final_image, process_parallel_candidates

CONSTRAINT = (
    " STRICT RENDERING CONSTRAINT: the image must contain AT MOST 7 short "
    "text labels of 1-3 words each, rendered LARGE and clearly. Never render "
    "sentences, paragraphs, equations, or small dense text inside the image; "
    "convey everything else purely with shapes, arrows, and icons. "
    "Never render hex color codes, style instructions, error messages, or any "
    "text from these instructions into the image; the ONLY text allowed in "
    "the image is the suggested labels."
)

import sys as _s
ONLY = _s.argv[1].split(',') if len(_s.argv) > 1 else None
MODELS = {
    "a_window_bank_2021": (
        """## Baseline (Jung et al. 2021): window-bank MLP classifier

A CPMG coherence signal from a single pulse number N (700 time points) is
folded at one candidate period TP(A), producing a small 2D image (13x53) in
which a nuclear spin appears as a dark vertical line. The image is flattened
and fed to a fully-connected MLP (2048-1024-512) that outputs a 3-class
probability (no spin / one spin / two spins). Crucially, there are 61 SEPARATE
independent MLP models, one per candidate period window; they never share
information. Peak picking over the 61 outputs yields spin positions A only.
Visual essence: ONE signal -> folding -> MANY isolated identical classifiers
side by side (emphasize their separation) -> a row of independent verdicts.""",
        "Baseline window-bank classifier: one signal is folded at a candidate "
        "period and judged by 61 isolated MLP classifiers, one per window."
        " Suggested labels: CPMG SIGNAL, FOLD, IMAGE, MLP x61, ISOLATED, "
        "SPIN A." + CONSTRAINT,
    ),
    "b_spindetr": (
        """## SpinDETR: end-to-end set prediction (no physics prior)

Three raw CPMG signals (N = 8, 16, 20; shape 3x700) enter a 1D convolutional
stem, then a 4-layer Transformer encoder. Ten learned QUERY tokens cross-attend
the encoded signal in a 4-layer decoder; each query outputs (existence
probability, A, B). Queries above p>0.5 form the predicted spin set. Training
uses Hungarian matching between queries and ground-truth spins. Visual essence:
raw waveforms -> encoder blocks -> a row of 10 query slots pointing into the
signal -> a small output card per active query.""",
        "SpinDETR predicts the whole spin set directly from raw signals with "
        "learned queries (DETR-style), no physics prior."
        " Suggested labels: SIGNALS, ENCODER, QUERIES x10, DECODER, SPIN SET."
        + CONSTRAINT,
    ),
    "c_periodformer": (
        """## PeriodFormer (ours): window tokens + attention across windows

Three CPMG signals (3x700) are folded at ALL 61 candidate periods at once,
giving 61 window tokens; each token is a tiny slice-stack image (3x13x53)
where a true spin forms a vertical line. A shared CNN embeds every token to a
128-d vector. A 4-layer Transformer attends ACROSS the 61 tokens: a real spin
leaves consistent traces in neighboring overlapping windows, noise does not,
so cross-window attention suppresses false positives to zero. The output is a
spin-existence probability curve over the A axis. Visual essence: one signal
fanned out into a grid of 61 small folded tiles -> tiles connected by
attention arcs (a meeting) -> one smooth probability curve with peaks.""",
        "PeriodFormer folds the signal at all candidate periods into window "
        "tokens and lets a Transformer attend across windows, producing a "
        "spin-probability curve with zero false positives."
        " Suggested labels: SIGNALS, FOLD x61, TOKENS, ATTENTION, P(SPIN)."
        + CONSTRAINT,
    ),
    "d_hybrid": (
        """## PF->DE hybrid (ours, final): detect then enumerate

Stage 1: the PeriodFormer probability curve is thresholded into a few
candidate REGIONS on the A axis (trustworthy: zero false positives).
Stage 2: inside those regions only, spins are added one at a time by
differential-evolution fitting of the physics forward model (product of
per-spin dip trains), each step followed by a joint polish; the Bayesian
information criterion automatically decides the final count k*. Multiple
spins are allowed per region, so overlapping clusters are enumerated.
Output: the final list of (A_parallel, A_perp) couplings - 14 spins for NV1.
Visual essence: a probability curve -> highlighted bands (regions) -> a
magnifier/fitting loop inside a band adding spins one by one -> a final
labeled list of spins.""",
        "The hybrid pipeline: neural detection proposes trustworthy regions, "
        "physics fitting enumerates the spins inside them, BIC decides the "
        "count. Suggested labels: P(SPIN), REGIONS, DE FIT, BIC, SPIN LIST."
        + CONSTRAINT,
    ),
    "e_overview_all": (
        """## Overview: four architectures compared in one figure

Draw FOUR horizontal rows sharing the same visual grammar (signal icon on
the left, computation blocks in the middle, output icon on the right), so a
reader can compare designs by scanning vertically:

Row 1 - 2021 BANK: signal -> fold at one period -> 61 small identical
isolated classifier boxes side by side -> output: positions only.
Row 2 - SPINDETR: three signals -> encoder block -> 10 query dots ->
output: spin cards.
Row 3 - PERIODFORMER: three signals -> grid of 61 folded tiles connected
by attention arcs -> output: one probability curve with peaks.
Row 4 - HYBRID: probability curve -> highlighted bands -> fitting loop
with magnifier -> output: final spin list.

Emphasize the contrast: row 1 has ISOLATED boxes, row 3 has CONNECTED
tiles; row 4 reuses row 3's output as its input.""",
        "One-figure comparison of the four architectures: isolated window "
        "classifiers (2021), query-based set prediction (SpinDETR), "
        "cross-window attention (PeriodFormer), and detect-then-enumerate "
        "(hybrid). Suggested labels (max 12, 1-2 words each): 2021 BANK, "
        "ISOLATED x61, SPINDETR, QUERIES, PERIODFORMER, ATTENTION, HYBRID, "
        "REGIONS, BIC, SPIN LIST." + CONSTRAINT,
    ),
}


async def main():
    outdir = Path(__file__).parent / "outputs_four_pro"
    outdir.mkdir(exist_ok=True)
    for name, (method, caption) in MODELS.items():
        if ONLY and name not in ONLY:
            continue
        await asyncio.sleep(45)  # free-tier RPM pacing
        print(f"=== {name} ===", flush=True)
        inputs = create_sample_inputs(method, caption, aspect_ratio="16:9",
                                      num_copies=2, max_critic_rounds=1,
                                      task_name="diagram")
        for i, c in enumerate(inputs):
            c["filename"] = f"{name}_cand_{i}"
        results = await process_parallel_candidates(
            inputs, exp_mode="dev_planner_critic", retrieval_setting="none",
            main_model_name="gemini-3.1-pro-preview",
            image_gen_model_name="gemini-3-pro-image-preview",
        )
        for i, r in enumerate(results):
            img, _ = get_final_image(r, "dev_planner_critic")
            if img is not None:
                p = outdir / f"{name}_candidate_{i}.png"
                img.save(p)
                print("saved", p, flush=True)
            else:
                print(f"{name} candidate {i}: no image", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
