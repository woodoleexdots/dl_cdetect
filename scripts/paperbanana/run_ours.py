"""Headless PaperBanana run for the NV-13C hybrid-pipeline figure."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import create_sample_inputs, get_final_image, process_parallel_candidates

METHOD = """
## Method: PF->DE hybrid pipeline for room-temperature detection of 13C nuclear spins

We detect individual 13C nuclear spins around a single NV center in diamond from
room-temperature CPMG dynamical-decoupling data. The input is a set of coherence
signals P_x(tau) measured with N = 8, 16, 20 pi-pulses (3 channels x 700 time
points). Each nuclear spin with hyperfine components (A_parallel, A_perp)
imprints periodic coherence dips with a characteristic local period TP(A).

Stage 1 - PeriodFormer (detection). The three signals are envelope-normalized,
then folded at all 61 candidate periods TP(A) for A in [-60, +60] kHz (2 kHz
grid), producing 61 "window tokens", each a small slice-stack image of shape
(3, 13, 53) in which a real spin appears as a vertical line. A shared CNN embeds
each token into a 128-d vector; a 4-layer Transformer attends ACROSS the 61
window tokens (real spins leave consistent traces in neighboring overlapping
windows, noise does not), and an existence head outputs a probability curve
P(spin) over the A grid. This stage is trained purely on synthetic scenes from
the physics forward model and produces zero false positives in benchmarks.

Stage 2 - region-constrained greedy DE (enumeration). The P(spin) curve is
thresholded into candidate A-regions. Inside these regions only, spins are
added one at a time by differential evolution fitting of the product forward
model M(tau) = prod_i M_i(tau; A_i, B_i), followed by joint L-BFGS polish;
the Bayesian information criterion selects the final spin count k*
automatically. Multiple spins are allowed per region, so clusters are
enumerated. The final output is the spin list {(A_parallel, A_perp)}_k*:
14 spins for sample NV1 and 10 spins for NV2, recovering all 7 manually
identified anchor spins.
"""

CAPTION = ("Overview of the proposed PF->DE hybrid pipeline: room-temperature "
           "CPMG signals are folded into period-window tokens, a Transformer "
           "attends across windows to produce a spin-existence curve, and "
           "region-constrained differential evolution with BIC enumerates the "
           "final list of 13C hyperfine couplings. "
           "STRICT RENDERING CONSTRAINT: the image must contain AT MOST 8 "
           "short text labels of 1-3 words each, rendered LARGE. Never render "
           "sentences, paragraphs, captions, equations, or small dense text "
           "inside the image; convey everything else purely with shapes, "
           "arrows, and icons.")


async def main():
    inputs = create_sample_inputs(
        METHOD, CAPTION, aspect_ratio="16:9", num_copies=2,
        max_critic_rounds=2, task_name="diagram",
    )
    results = await process_parallel_candidates(
        inputs, exp_mode="dev_planner_critic", retrieval_setting="none",
        main_model_name="gemini-3.5-flash",
        image_gen_model_name="local/qwen-image",
    )
    outdir = Path(__file__).parent / "outputs_ours"
    outdir.mkdir(exist_ok=True)
    for i, r in enumerate(results):
        img, desc = get_final_image(r, "dev_planner_critic")
        if img is not None:
            p = outdir / f"hybrid_pipeline_candidate_{i}.png"
            img.save(p)
            print("saved", p, flush=True)
        else:
            print(f"candidate {i}: no image", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
