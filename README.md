# MA2015 lessons

Public, reusable lessons for MA2015 (Bioinspired Mathematical Algorithms).
The repository contains academic lesson material in English and Spanish:
slides, source code, figures, notebooks, and student-facing documentation.
It has no student records, private course operations, or instructor-only notes.

## Use

```text
uv sync --frozen
uv run es/Leccion_01_Intro/src/introduccion_01_el_paisaje.py
```

The LaTeX templates used by the decks are in `templates/en` and
`templates/es`, so each deck compiles without files from a parent repository.
PDFs and notebooks are tracked with Git LFS.

The original course repository includes this repository at `02-lecciones` as a
Git submodule.

