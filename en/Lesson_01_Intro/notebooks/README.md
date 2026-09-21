# Lesson 01 - Guided notebooks

The published classroom route is the [compact notebook](lesson_01_intro_compact.ipynb).
The [complete notebook](lesson_01_intro.ipynb) remains the reference version with
detailed code-reading notes and optional experiments.

## Compact classroom route

The compact version follows the seven canonical scripts in 29 cells, including 9
executable cells. It keeps the mathematical model, prediction prompts, seven
figures, expected results, final assertions, and local/VS Code/Colab path notes.
It removes the extended code-reading traces and optional experiments from the
main 50-minute route.

Open [lesson_01_intro_compact.ipynb](lesson_01_intro_compact.ipynb) for the
linear classroom route. Open [lesson_01_intro.ipynb](lesson_01_intro.ipynb) for a complete, incremental
walkthrough of your first genetic algorithm. The notebook includes explanations,
complete code, saved outputs, seven figures, prediction questions, self-check
answers, and optional experiments. Every code cell has a nearby reading guide
for its purpose, inputs, operations, outputs, and later use. Allow about 50
minutes for the main walkthrough; detailed reading and optional experiments can
continue afterward.

## Local Jupyter or VS Code

1. From the repository root, run `uv sync --frozen` once.
2. Open `lesson_01_intro_compact.ipynb` for class, or `lesson_01_intro.ipynb`
   for the complete reference, in Jupyter or VS Code.
3. In VS Code, choose **Select Kernel** and select the project's `.venv` /
   `ma2015-genetic-algorithms` Python environment.
4. Restart the kernel and choose **Run All**. The project requires Python 3.11+
   and already declares NumPy, Matplotlib, and `ipykernel`.

When VS Code exposes the notebook's local path, generated PNGs go into
`notebooks/figures/` beside the notebook even if the kernel starts from the
workspace root. Other local Jupyter frontends fall back to their current
working directory, so open the notebook from this directory when practical.

## Google Colab extension in VS Code

1. Install Google's official [Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab).
2. Open the local notebook in VS Code.
3. Choose **Select Kernel → Colab → Auto Connect** and complete Google sign-in
   when prompted.
4. Choose **Run All**. A standard CPU server is sufficient; no repository
   upload, Drive mount, external data, or accelerator is required.

The notebook file remains local while its code runs on the remote Colab kernel.
Because that kernel cannot see the host notebook path, generated PNGs are placed
in the remote working directory, normally `/content/figures`. Retrieve them
through the Colab activity bar's **Contents** view if needed. The figures also
remain embedded in the local notebook when it is saved. See Google's current
[extension user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## Google Colab in a browser

1. Download the notebook and open [Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook**, then select the downloaded file.
3. Save a personal copy in Drive to keep edits. Use a standard CPU runtime.
4. Run cells from top to bottom with **Shift+Enter**. After changing earlier
   definitions, rerun dependent cells; restart the runtime and run all cells
   if execution state becomes confusing.
5. Save or download your notebook to retain code, notes, and embedded outputs.

No repository checkout, mounted Drive, external data, or slide images are needed.
The computation uses Python's standard library, NumPy, and Matplotlib, which a
standard Colab CPU runtime provides. Colab's filesystem is temporary; save the
notebook to retain its embedded figures.

## Relationship to the lesson

The compact notebook follows the [compact slides](../slides/lesson_01_compact.tex)
and the [seven standalone scripts](../src/) in the same order. The complete
notebook follows the [complete slides](../slides/lesson_01.tex) and the same
seven standalone scripts:

| Step | Main addition |
|---|---|
| 1 | Objective function and landscape |
| 2 | Individual, random candidate, and population |
| 3 | Tournament selection |
| 4 | Clamping and blend crossover |
| 5 | Gaussian mutation |
| 6 | Generational loop and convergence history |
| 7 | Seed 16 and the local-optimum comparison |

Definitions accumulate across cells; both notebooks are self-contained as a whole.
Each change recipe has matching `NEW` or `CHANGED` markers. Step 6 wraps the
existing loop in `run_ga(seed)`, and step 7 reuses it. Settings, operator logic,
and random draw order match the scripts. Short concrete traces expose one
tournament and the list mechanics of adjacent pairing, flattening offspring,
mutation replacement, and population replacement. The notebook clarifies grid
approximations, finite mutation samples, crossover reach, cached fitness, object
identity, and the seed's effect on the whole random sequence.

Keep the baseline settings for the first walkthrough. The optional experiments
use separate variables and do not overwrite the main results. The **Checks**
cell verifies the documented baseline numbers and repeated-run reproducibility.

## Validation

Verified locally on date omitted with Python 3.11.16, NumPy 2.4.6, and
Matplotlib 3.11.1 from the course lockfile. Notebook tooling (`ipykernel`,
`nbformat`, and `nbconvert`) is included in the locked project environment.

- Validated the notebook format and executed all 104 cells (34 code cells) in a
  fresh kernel, starting in an isolated directory containing only the notebook.
- Audited every executable cell for an immediately adjacent explanation; the
  supporting setup, reporting, plotting, assertions, and optional experiments
  receive the same reading support as the genetic-algorithm operators.
- Retained all seven inline figures and executed outputs, including the optional
  experiments. Repeated the population, selection, crossover, and mutation
  demonstrations; the same seeds reproduced the same baseline results.
- Confirmed 4/10 unselected individuals, 14/20 crossover children outside the
  parental interval, and mutation arrivals of 0/2000 and 110/2000. Both full runs
  matched the original scripts' entire generation histories and final gene
  values exactly.
- Checked population size, gene bounds, fitness consistency, recipe markers,
  and repeated full runs. The existing script band checker passed.
- Inspected the seven figures and the HTML-rendered guide, equations, and
  vocabulary table. Checked navigation and text overflow at desktop and
  narrower reading widths.

This is local Jupyter execution and browser validation; the complete notebook has not
been executed inside Google's hosted Colab service or through the VS Code Colab
extension. Its remote-kernel path handling was validated separately with an
unavailable host path and an isolated working directory.

The compact notebook was verified on September 22, 2026 with 29 cells, 9 code
cells, zero errors, seven embedded figures, and the documented baseline results.
