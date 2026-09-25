# Lesson 01 — Your First Genetic Algorithm

Build a genetic algorithm in seven autonomous increments, then compare two seeded runs. The prerequisites are basic Python variables, conditionals, loops and functions. Objects, arrays, comprehensions, pairing and plotting are explained where they appear.

## Materials

- [Extensive deck — study](slides/lesson_01.pdf), 38 pages; [LaTeX source](slides/lesson_01.tex).
- [Extensive notebook — study](notebooks/lesson_01_intro.ipynb): full foundations, incremental code, saved results, exercises and expandable solutions.
- [Compact deck — 40-minute class](slides/lesson_01_compact.pdf), 19 pages; [LaTeX source](slides/lesson_01_compact.tex).
- [Compact notebook — 40-minute class](notebooks/lesson_01_intro_compact.ipynb): complete prepared code and brief checks.
- [Seven independent scripts](src/) and [generated figures](figures/).
- [Student package](lesson_01_student_package.zip): extensive PDF/notebook, scripts, figures, instructions and the course dependency lock. Compact teaching materials are separate.
- [Execution instructions](notebooks/README.md).

## Model and evidence

The synthetic, dimensionless model maximizes `sin(x) - 0.2*abs(x)` on `[-10,10]`, with radians. It is a teaching example, not measured Planta Física data. Random populations and variation are part of the learning objective; there is no external dataset or auxiliary data-preparation step. The extensive materials derive the analytic global maximizer `acos(0.2)` and distinguish it from both the sampled grid and the algorithm's best observation.

| Step | Added idea | Observed result and check |
|---|---|---|
| 1 | Objective and landscape | Three interior peaks and a partial boundary hill; grid best `x=+1.378`, `f=+0.706`. A grid is an approximation. |
| 2 | Individual and population | Ten candidates; best initial `x=-0.323`, `f=-0.382` for seed 52. Check the cached score. |
| 3 | Tournament selection | Four original individuals have zero copies. Selection changes multiplicity and creates no genes. |
| 4 | Blend crossover and clamp | 14 of 20 children lie outside the parental interval; all satisfy the domain. The two proposals share a draw. |
| 5 | Gaussian mutation | From -4.6, 0/2000 and 110/2000 arrivals satisfy `abs(x-1.38)<1.5` for sigma 1 and 3. Zero observations do not imply zero probability. |
| 6 | Complete replacement loop | Seed 52 gives `x=+1.372`, `f=+0.706` after ten generations. Check population size, bounds and cached fitness. |
| 7 | Change seed to 16 | Final `x=-4.417`, `f=+0.073`. The seed changes the entire random sequence; two runs do not estimate a success rate. |

These displayed values come from the scripts. Reproducibility comparisons use complete ordered populations and histories, not rounded summaries. Counts and within-environment results are exact. Across runtimes, compare floating-point values at `rtol=atol=1e-12`; investigate changed trajectories.

## How the code is marked up

Two devices, both mandatory course-wide, both already
applied here:

- **The recipe** — every script after the first opens with a `CHANGES FROM …`
  block in its docstring: an ordered list of the changes that turn the previous
  script into this one. That order is the order to type them in class, and the
  order the slides must follow.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band in
  the body, at the site of the change, numbered and named to match the recipe.
  Everything outside a band is code the students already have. Step 7 changes a
  value instead of adding code, so it marks the line: `SEED = 16   # --- CHANGED ---`.

A slide for step *n* is therefore mechanical to write: it is the recipe of
script *n*, in order, with the banded code as the listings.

Each script also explains its supporting Python locally, so it can be read on
its own. Comments and focused docstrings cover imports, paths, random state,
arrays, object references, comprehensions, formatting, validation, and plotting.
The slides keep the seven-step classroom route concise and collect these
language notes in a clearly marked code-reading appendix.

## Execution and provenance

From the MA2015 repository root:

```sh
uv sync --locked
uv run python 02-lecciones/en/Lesson_01_Intro/src/first_example_01_the_landscape.py
```

Each script runs without arguments or imports from other steps, saves its PNG relative to its own location and closes figures. With a headless backend, use `MPLBACKEND=Agg`. The notebook saves figures relative to the kernel working directory and embeds them in its outputs.

The academic source is Ivan Gridin, *Learning Genetic Algorithms with Python*, Chapter 1. The course preserves its algorithmic sequence while adding independent increments and explanatory material. See the extensive materials for derivations, syntax examples and the connection to engineering design decisions.

## 2026-09-25 · compact code guidance

Every code cell in the compact instructor notebook now carries explanatory comments beside the relevant operation. The executable Python and saved outputs were preserved; a fresh run reproduced the saved outputs. Student materials were not changed.

| Notebook | Code cells | SHA-256 |
|---|---:|---|
| [lesson_01_intro_compact.ipynb](notebooks/lesson_01_intro_compact.ipynb) | 7 | 5543AEBB830CEDA86A3D5C63E9B78EEC2F8C6A69C5C9A6AFC8E57A0A7E93C89A |
