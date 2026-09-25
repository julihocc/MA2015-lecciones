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

## 2026-09-25 · compact code guidance

Every code cell in the compact instructor notebook now carries explanatory comments beside the relevant operation. The executable Python and saved outputs were preserved; a fresh run reproduced the saved outputs. Student materials were not changed.

| Notebook | Code cells | SHA-256 |
|---|---:|---|
| [leccion_01_intro_compacta.ipynb](es/Leccion_01_Intro/docente/notebooks/leccion_01_intro_compacta.ipynb) | 7 | 0415662E2DC36935A27646EC3EA9AEA79E38E22386E47712667C4AAE8E70F577 |
| [lesson_01_intro_compact.ipynb](en/Lesson_01_Intro/notebooks/lesson_01_intro_compact.ipynb) | 7 | 5543AEBB830CEDA86A3D5C63E9B78EEC2F8C6A69C5C9A6AFC8E57A0A7E93C89A |
