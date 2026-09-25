# Lesson 01 — Notebooks

- [Extensive — study](lesson_01_intro.ipynb): foundations, complete incremental code, optional experiments and explained solutions.
- [Compact — 40-minute class](lesson_01_intro_compact.ipynb): prepared code, results and short checks; its opening table matches the compact deck.

## Local execution

1. From the MA2015 repository root, or the extracted student-package root, run `uv sync --locked`.
2. Open the notebook in Jupyter or VS Code and select that environment's Python kernel.
3. Restart the kernel and run all cells in order. Python 3.11 or later is required.
4. Save the notebook to retain outputs. PNGs also go to `figures/` relative to the kernel's working directory. No absolute host paths are used.

## Google Colab

Upload the notebook at [Google Colab](https://colab.research.google.com/), select a CPU runtime, and run all cells in order. Save a personal copy or download the executed notebook. No external dataset, repository checkout or mounted Drive is needed. The remote filesystem is temporary.

## Interpretation and reproducibility

Keep baseline parameters and seeds for the first run. The extensive notebook explains each cell and uses separate names for optional experiments. Both versions retain seven inline figures. The checks cover population size, legal genes, cached fitness, counts and repeated seeded histories. A seed controls every subsequent random draw.

The script's mutation figure shows an unbounded proposal density; the notebook shows a histogram of sampled, clipped proposals. Both use the same samples and arrival criterion. Neither a density height nor a finite observed fraction is a guaranteed probability for another run.

The current revision has passed two isolated local executions per notebook. Both notebooks also passed Run All on new hosted Google Colab CPU runtimes on 22 September 2026 (Python 3.13.15, NumPy 2.1.3, Matplotlib 3.10.0), including the final baseline and reproducibility assertions.

## 2026-09-25 · compact code guidance

Every code cell in the compact instructor notebook now carries explanatory comments beside the relevant operation. The executable Python and saved outputs were preserved; a fresh run reproduced the saved outputs. Student materials were not changed.

| Notebook | Code cells | SHA-256 |
|---|---:|---|
| [lesson_01_intro_compact.ipynb](lesson_01_intro_compact.ipynb) | 7 | 5543AEBB830CEDA86A3D5C63E9B78EEC2F8C6A69C5C9A6AFC8E57A0A7E93C89A |
