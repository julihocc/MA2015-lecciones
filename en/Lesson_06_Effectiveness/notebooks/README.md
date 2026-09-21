# Lesson 06 � guided notebooks

These notebooks are the self-contained, student-facing path through the lesson's numbered script sequences. Use each notebook from top to bottom; every executable cell has a reading guide immediately before it.

## Index

- [`lesson_06_success_rate.ipynb`](lesson_06_success_rate.ipynb) � why one seeded run is not a success-rate estimate. 33 cells, 11 semantic outputs, 4 inline figures.

## Run locally

From the repository root, run `uv sync --frozen`, select `.venv` as the Jupyter or VS Code kernel, open the notebook, then **Restart Kernel and Run All**. The notebook writes temporary outputs beside its runtime copy and needs no repository import or external data file.

## Run in Colab

Upload the notebook to browser Colab, or open it with the VS Code Colab extension, save a personal copy, and run all cells in order. Lesson 11 installs pinned `python-igraph==1.0.0` only if importing it fails. Generated figures exist only for that runtime unless you download them.

## Scope and validation

The frozen scripts remain the canonical compact examples; the notebooks add guided reading, predictions, interpretations, optional experiments, answers, and final assertions. On date omitted each notebook passed `nbformat` validation, code compilation, unique-ID and guide-adjacency checks, isolated execution with only the notebook present, and three fresh-kernel path simulations: generic local, accessible VS Code-style path, and inaccessible remote-style path. Retained results and figure filenames matched the scripts, and the paired Spanish notebook matched the same deterministic numerical evidence. The locked environment used Python 3.11.16, NumPy 2.4.6, Matplotlib 3.11.1, pandas 3.0.5, nbformat 5.11.1, and nbconvert 7.17.1.

Authenticated VS Code Colab-extension and browser-Colab runs were not tested; path simulation is not a hosted Colab test.
