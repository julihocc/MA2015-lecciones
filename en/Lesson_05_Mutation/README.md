# Lesson 05 — Mutation

Mutation is the only operator that can produce a value the population never had,
so the only question worth asking about it is how far one application can reach.
This lesson builds an instrument that measures reach, applies it to all seven of
Gridin's mutation operators across two encodings, and takes it back to the run
Lesson 02 left for dead — where it finds that mutation was never what was
missing.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 5 (*Mutation*) — 7 sections, `1,895 words, 7 figures, 113 lines across
  7 files (`bit_flip.py`, `exchange.py`, `inversion.py`, `shift.py`,
  `shuffle.py`, `random_deviation.py`, `fitness_driven.py`). One file per
  operator, and no measurement of any of them.
- **Budget:** 35 min (Chapter 5 is 5.4% of the book — the lightest of the four
  operator chapters). Six steps, and the sequence fits: the whole thing runs in
  about **10 s**.
- **Prerequisites:** Lessons 01 and 02. Step 1 of the first example reproduces
  Lesson 01 step 5 to the digit, and step 4 reproduces Lesson 02 step 5 to the
  digit; both identities are printed by the scripts themselves.

## Mathematical model

For a length-\(L\) real chromosome, each gene is activated by
\(B_i\sim\operatorname{Bernoulli}(p)\) and receives
\(\Delta_i\sim\mathcal N(\mu,\sigma^2)\), followed here by projection to the
legal interval. The expected number of activated genes is \(Lp\); \(\sigma\)
controls displacement, not activation. Hamming displacement is
(H(x,x')=\sum_i I[x_i\ne x_i']); for permutations, total index travel is
(D(\pi,\pi')=\sum_v|\operatorname{pos}_\pi(v)-\operatorname{pos}_{\pi'}(v)|).
Permutation mutations preserve membership. A
zero-width top step has probability zero under an ideal continuous proposal.

## What the student leaves with

1. **Mutation has one idea — reach — and two dials that set it.** `sigma` says
   how far a touched gene moves; `p` says how many genes are touched. Their
   product is the movement budget; `p` alone decides whether it is spent as one
   long jump or as ten short ones, and only the long jump can cross a valley.
2. **Reach cannot simply be bought.** Bigger is a direction, not an improvement:
   the small sigma finds the most uphill moves and never arrives anywhere, the
   large sigma arrives and cannot stay.
3. **Fitness-driven mutation wins per mutation and loses per fitness call.**
   Retrying until you improve adds no new ability — it buys more tickets in the
   same lottery, at three times the price.
4. **The encoding decides which mutations exist.** On a permutation, adding
   noise is not a weak mutation, it is an invalid one; the only legal moves are
   rearrangements, and their reach has to be measured in two numbers, not one.
5. **Before tuning an operator, check that the target exists.** Lesson 02's
   staircase could not be solved by any mutation regime because its top step had
   zero width. One run cannot tell a dead search from an impossible one.

## The running examples

Two, because Chapter 5's operators split cleanly in two and forcing them into one
sequence would hide the split that matters.

- **`mutation_reach_`** — a real-valued gene on Lesson 01's landscape
  `f(x) = sin(x) − 0.2·|x|`, and later Lesson 02's staircase. Four steps.
- **`permutation_mutation_`** — a chromosome that is an *order* of ten symbols,
  where a gene has no value to perturb. Two steps.

One measuring idea runs through both: fire the operator 2,000 times at one fixed
individual from one fixed seed and report the distribution, not an example. The
book prints one before-and-after pair per operator; that is what this lesson
replaces.

The permutation half is two steps rather than five because the chapter's four
permutation operators are 40 lines between them and padding them would cost the
budget the first half needs.

## The steps

### First example — `mutation_reach_` (real-valued genes)

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `mutation_reach_01_the_instrument.py` | `reach_report()`, the displacement distribution | Lesson 01's two numbers reappear exactly — sigma 1.0 arrives **0 of 2000**, sigma 3.0 **110 of 2000**. Mean travel is 0.798·sigma until the walls start eating it: at sigma 6.0, **19.4%** of proposals are clamped and the ratio drops to 0.690. The trade in one table: sigma 0.5 finds the most uphill moves (**272**) and arrives **0** times |
| 2 | `mutation_reach_02_rate_and_step.py` | `GENE_COUNT`, `mutate_random_deviation()`, `regime_report()` | Genes touched never differs from `p·n` by more than **0.015**. Three regimes with the same `p·sigma = 0.30` travel 1.904 / 2.179 / 2.304 in total but their longest single step differs by a factor of **7.1** (10.253 vs 1.449). At p = 0.1, **718 of 2000** mutations change nothing at all, against a predicted (1−p)¹⁰ = 0.3487 |
| 3 | `mutation_reach_03_fitness_driven.py` | `MAX_TRIES`, `mutate_fitness_driven()`, `compare()` | The filter turns a mean fitness change of **−1.0337 into +0.0548** and doubles arrivals (102 → 273) — but spends **5,603 fitness calls against 2,000**, so per thousand calls it arrives **48.7** times against blind mutation's **51.0**. Acceptance 20.2% is exactly 1−(1−q)³ for the blind uphill rate q |
| 4 | `mutation_reach_04_the_dead_staircase.py` | `run()`, `staircase()`, `step_widths()`, `PEAK_HEIGHT_REPAIRED`, `sweep()` | Lesson 02's run reproduced (6 generations, best **+1.8000**, spread 0.192, **0.2000** short). Then eight mutation regimes × 100 runs: the top step is reached **0 times in 800 runs**, while mean gene spread goes from 0.462 to 3.194. `step_widths()` says why — the top step is **0.000 wide**. Repaired (peak height 10.0 → 10.5, top step 0.500 wide), Lesson 02's own regime reaches it **96 of 100** times |

### Second example — `permutation_mutation_` (an order, not a vector)

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `permutation_mutation_01_illegal_noise.py` | `is_permutation()`, `travel()`, `mutation_bit_flip()`, `mutation_exchange()`, `rearrangement_report()` | Random deviation on a permutation produces **1935 invalid mutants of 2000**, and **0** of the 65 survivors had any gene touched — the operator's only legal output is the untouched one, at a measured rate matching (1−p)¹⁰ to within **0.0025**. Bit flip changes exactly 1 position every time; exchange exactly 2 |
| 2 | `permutation_mutation_02_rearrangements.py` | `mutation_inversion()`, `mutation_shift()`, `mutation_shuffle()`, `compare_operators()` | The two measurements of reach rank the operators differently: by positions disturbed **shift (4.663) > inversion (4.230) > shuffle (3.708) > exchange (2.000)**, by distance travelled **inversion (1.308) > shuffle (0.876) > exchange (0.733) = shift (0.733)**. Exchange and shift give identical total travel on **90 of 90** position pairs. Shuffle returns the individual unchanged **271 of 2000** times (13.6%) |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these. Re-run the script
rather than trusting this table if the code has changed.

**Suggested teaching order:** `mutation_reach_` 01→04, then
`permutation_mutation_` 01→02. The second example's last step closes both.
Fitness-driven mutation inside a run, priced per evaluation, is Lesson 06.
A travelling-salesperson search that uses these permutation operators is
Lesson 10; this lesson stops at what they do to one chromosome.

## A finding worth keeping — and a defect it exposes in Lesson 02

Step 4 was designed to redeem a promise. Lesson 02 step 5 ended with a run that
died on a staircase, 0.2000 short of the optimum with no gene spread left, and
said mutation "is the one operator that could put it back". The plan was to show
a different mutation regime reviving it.

It does not. Eight regimes × 100 runs each reach the top step **zero times**,
while multiplying the surviving gene spread by 6.9. A refusal that flat is not
about tuning, so the step asks the other question — and finds that Lesson 02's
staircase,

```
floor((10.0 − 2·|x − 4|) / 1.0) · 0.2
```

has a **top step of zero width**. The tent peaks at exactly 10.0, so the topmost
`floor()` band is the single point `x = 4.0`. The brute-force check in Lesson 02
evaluated on `numpy.linspace(-10, 10, 20001)`, whose spacing is 0.001 and which
therefore includes `x = 4.0` exactly; that is the only reason the optimum was
reported as +2.0000. A gene drawn from a continuous distribution lands there with
probability zero. **Lesson 02's published 0.2000 shortfall is unreachable by
construction, and no operator could ever have closed it.**

Raising the tent by half a step (`PEAK_HEIGHT_REPAIRED = 10.5`) gives the top
step a width of 0.5 without changing the number of steps, the rise, the slope, or
the fact that the peak is interior. On that landscape Lesson 02's own regime —
p = 0.1, sigma = 1.0, the one it called insufficient — reaches the top in **96 of
100 runs**.

The step teaches that instead, which is more than the planned version would have.

## How the code is marked up

Two devices, mandatory course-wide:

- **The recipe** — each script after the first of its prefix opens with a
  `CHANGES FROM …` block listing, in order, the changes that turn the previous
  script into this one. That order is the order to type in class and the order
  the slides follow.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band at
  the site of the change, numbered and named to match. Everything outside a band
  is code the students already have.
  `mutation_reach_02_rate_and_step.py` changes one line instead of adding code
  for its second item, so it marks the line:
  `self.fitness = …   # --- CHANGED --- a vector is scored by summing over its genes`.

en/Lesson_05_Mutation/src` — every line must read `OK`, except the two
first-of-prefix scripts, which have no recipe and read `---`.

## Running it

```bash
uv run en/Lesson_05_Mutation/src/mutation_reach_01_the_instrument.py
```

Each script is self-contained, takes no arguments, and saves its figure to
`../figures/` under its own name. The six scripts run in about **10 s** in total
(1.3 – 2.2 s each); 2,000 draws per measurement and 100 runs per regime are what
buy the stable rates.

## Folder contents

```
Lesson_05_Mutation/
├── README.md      this file — the lesson's contract
├── src/           the six numbered scripts, two prefixes
├── figures/       one figure per script
```

## Active full package — W08, 25 September 2026

Five components: 6 scripts in src/, extensive deck and notebook in student/, compact deck and notebook in instructor/. 6 figures in figures/. Both PDFs compiled twice; notebooks and scripts ran without errors. The compact deck has nine slides and comments in every code cell.

Forty-minute route per lesson. In S04: 51 minutes for L04, 35 for L05, and 14 for transitions and closure. Synthetic examples remain distinct from the Planta Física model. External data and ZIP: not applicable. Hosted Colab, full accessibility, and student distribution were not checked.

## Guided notebooks

The current complete student notebook is in [`student/`](student/). The earlier milestone notebooks remain in [`notebooks/`](notebooks/) as supplementary material.
