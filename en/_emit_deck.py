"""Emit a Lesson-01-style Beamer deck from a lesson folder.

Listings are \\lstinputlisting of the script on disk (never pasted code).
Captions come from the lesson README table. No TODO-run frames.

Usage, from the course root:

    python en/_emit_deck.py en/Lesson_02_GA_Flow
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from _bridges import closer_frame, intro_frame

NEW_BAND = re.compile(r"^\s*# --- NEW \((\d+)\)\s+(.+?)\s+---")
CHANGED_ANY = re.compile(r"# --- CHANGED ---")
CHANGED_BLOCK = re.compile(r"^\s*# --- CHANGED ---")
CLOSING = re.compile(r"^\s*# -{3,}\s*$")
RECIPE_ITEM = re.compile(r"^\s+(\d+)\.\s+(.+)$")
TITLE_LINE = re.compile(
    r"^Lesson\s+\d+\s+-\s+(?:Step\s+\d+|[\w ]+?\s+\d+):\s+(.*)$"
)
SCRIPT_IN_CELL = re.compile(r"`([^`]+\.py)`")


def latex_escape(text: str) -> str:
    """Escape prose for a Beamer frame. Keep it readable, not perfect math."""
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    # Markdown leftovers in README claims.
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


def frame_title_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def numbered_scripts(src: Path) -> list[Path]:
    files = [
        p
        for p in src.glob("*.py")
        if re.search(r"_\d{2}_", p.name)
    ]
    return sorted(files)


def docstring_of(text: str) -> str:
    if '"""' not in text:
        return ""
    return text.split('"""', 2)[1]


def step_title(doc: str, stem: str) -> str:
    for line in doc.splitlines():
        match = TITLE_LINE.match(line.strip())
        if match:
            return match.group(1).strip()
    return stem.replace("_", " ")


def question_prose(doc: str) -> list[str]:
    """Opening paragraphs of the docstring, before the recipe."""
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if TITLE_LINE.match(stripped) or stripped.startswith("===="):
            i += 1
            continue
        if stripped.startswith("NEW IN THIS STEP"):
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            continue
        if stripped == "":
            i += 1
            continue
        break
    body: list[str] = []
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("CHANGES FROM") or stripped.startswith("Run it:"):
            break
        if stripped.startswith("REMOVED FROM"):
            break
        body.append(lines[i].rstrip())
        i += 1
    while body and body[-1] == "":
        body.pop()
    paragraphs: list[list[str]] = [[]]
    for line in body:
        if line.strip() == "":
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        paragraphs[-1].append(line.strip())
    paragraphs = [p for p in paragraphs if p]
    kept = paragraphs[:1]
    out: list[str] = []
    for n, para in enumerate(kept):
        if n:
            out.append("")
        out.extend(para)
    return out


def recipe_label(rest: str) -> str:
    """Left column of a recipe line: the name, not the trailing gloss."""
    if "  " in rest:
        return rest.split("  ", 1)[0].strip()
    return rest.strip()


def recipe_items(doc: str) -> list[tuple[int, str]]:
    items = []
    in_recipe = False
    for line in doc.splitlines():
        if "CHANGES FROM" in line or "Introduce them in this order" in line:
            in_recipe = True
            continue
        if in_recipe and line.strip().startswith("Run it:"):
            break
        if in_recipe:
            match = RECIPE_ITEM.match(line)
            if match:
                items.append((int(match.group(1)), recipe_label(match.group(2))))
    return items


def band_ranges(text: str) -> dict[int, tuple[int, int, str]]:
    """Map NEW band number -> (firstline, lastline, name). 1-based lines."""
    lines = text.splitlines()
    found: dict[int, tuple[int, int, str]] = {}
    i = 0
    while i < len(lines):
        match = NEW_BAND.match(lines[i])
        if not match:
            i += 1
            continue
        number = int(match.group(1))
        name = match.group(2).strip()
        start = i
        end = i
        j = i + 1
        while j < len(lines):
            if NEW_BAND.match(lines[j]) or CHANGED_BLOCK.match(lines[j]):
                end = j - 1
                break
            if CLOSING.match(lines[j]):
                end = j
                break
            j += 1
        else:
            end = len(lines) - 1
        found[number] = (start + 1, end + 1, name)
        i = end + 1
    return found


def changed_ranges(text: str) -> list[tuple[int, int]]:
    """1-based inclusive ranges for each CHANGED marker, in file order."""
    lines = text.splitlines()
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not CHANGED_ANY.search(line):
            i += 1
            continue
        start = i
        if CHANGED_BLOCK.match(line) or (
            line.lstrip().startswith("#") and CHANGED_ANY.search(line)
        ):
            j = i + 1
            seen_code = False
            while j < len(lines):
                nxt = lines[j]
                if NEW_BAND.match(nxt) or CHANGED_BLOCK.match(nxt):
                    break
                if CLOSING.match(nxt) and seen_code:
                    j += 1
                    break
                stripped = nxt.strip()
                if stripped.startswith("print(") and seen_code:
                    break
                if stripped and not stripped.startswith("#"):
                    seen_code = True
                j += 1
            end = j - 1
        else:
            end = i
        ranges.append((start + 1, end + 1))
        i = end + 1
    return ranges


def listing_plan(text: str, recipe: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
    """Ordered (frame title, firstline, lastline) for the recipe."""
    news = band_ranges(text)
    changed = changed_ranges(text)
    changed_i = 0
    plan: list[tuple[str, int, int]] = []
    for number, name in recipe:
        if number in news:
            first, last, band_name = news[number]
            title = f"({number}) {band_name}"
            plan.append((title, first, last))
            continue
        if changed_i < len(changed):
            first, last = changed[changed_i]
            changed_i += 1
            title = f"({number}) {name}"
            plan.append((title, first, last))
    return plan


def readme_claims(readme: Path) -> dict[str, str]:
    claims: dict[str, str] = {}
    if not readme.exists():
        return claims
    for line in readme.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"#", "---"}:
            continue
        script = None
        claim = None
        for cell in cells:
            match = SCRIPT_IN_CELL.search(cell)
            if match:
                script = match.group(1)
        if script is None:
            continue
        # Last non-script cell is the verified claim.
        for cell in reversed(cells):
            if SCRIPT_IN_CELL.search(cell):
                continue
            if re.fullmatch(r"\d+", cell):
                continue
            if cell in {"Script", "What it adds", "What its output proves",
                        "Verified result"}:
                continue
            claim = cell
            break
        if claim:
            claims[script] = claim
    return claims


def lesson_heading(lesson_dir: Path) -> tuple[str, str]:
    parts = lesson_dir.name.split("_")
    number = parts[1]
    title = " ".join(parts[2:]).replace("_", " ")
    return number, title


def emit_tex(lesson_dir: Path) -> str:
    src = lesson_dir / "src"
    figures = lesson_dir / "figures"
    claims = readme_claims(lesson_dir / "README.md")
    number, title = lesson_heading(lesson_dir)
    chunks: list[str] = [
        r"\input{../../../00-templates/preamble.tex}",
        "",
        r"% Listings pulled from src/ with \lstinputlisting, so the slide is the file.",
        r"\lstset{",
        r"    basicstyle=\ttfamily\fontsize{8pt}{9.6pt}\selectfont,",
        r"    breaklines=true,",
        r"    breakatwhitespace=true,",
        r"    columns=fullflexible,",
        r"    keepspaces=true,",
        r"    xleftmargin=0.3em,",
        r"    framesep=2pt,",
        r"    aboveskip=0pt,",
        r"    belowskip=0pt",
        r"}",
        "",
        r"\newcommand{\resultframe}[3]{%",
        r"    \begin{frame}{#1}",
        r"        \begin{center}",
        r"            \includegraphics[height=0.62\textheight,width=\textwidth,keepaspectratio]{#2}",
        r"        \end{center}",
        r"        \vspace{0.25em}",
        r"        {\small #3\par}",
        r"    \end{frame}%",
        r"}",
        "",
        r"\newcommand{\claimframe}[2]{%",
        r"    \begin{frame}{#1}",
        r"        {\small #2\par}",
        r"    \end{frame}%",
        r"}",
        "",
        r"\date{}",
        "",
        r"\begin{document}",
        "",
        f"\\lessontitle{{{number}}}{{{title}}}",
        "",
    ]

    chunks.append(intro_frame(f"{int(number):02d}"))
    chunks.append("")

    for script in numbered_scripts(src):
        text = script.read_text(encoding="utf-8")
        doc = docstring_of(text)
        heading = step_title(doc, script.stem)
        prose = question_prose(doc)
        recipe = recipe_items(doc)
        listings = listing_plan(text, recipe)
        claim = claims.get(script.name, "")
        fig = figures / f"{script.stem}.png"
        rel_script = f"../src/{script.name}"
        rel_fig = f"../figures/{script.stem}.png"

        chunks.append(f"% ================= {script.stem} =================")
        chunks.append(f"\\section{{{frame_title_escape(heading)}}}")
        chunks.append("")
        chunks.append(f"\\begin{{frame}}{{{frame_title_escape(heading)}}}")
        if prose:
            for line in prose:
                if line == "":
                    chunks.append("")
                    continue
                chunks.append(latex_escape(line))
        else:
            chunks.append(latex_escape(heading))
        chunks.append(r"\end{frame}")
        chunks.append("")

        for title_raw, first, last in listings:
            n_lines = last - first + 1
            chunks.append(r"\begin{frame}[fragile]{" + frame_title_escape(title_raw) + "}")
            style = r"basicstyle=\ttfamily\fontsize{8pt}{9.6pt}\selectfont"
            if n_lines > 28:
                style = r"basicstyle=\ttfamily\fontsize{6pt}{7.2pt}\selectfont"
            elif n_lines > 18:
                style = r"basicstyle=\ttfamily\fontsize{7pt}{8.5pt}\selectfont"
            chunks.append(
                f"\\lstinputlisting[{style},"
                f"firstline={first},lastline={last}]{{{rel_script}}}"
            )
            chunks.append(r"\end{frame}")
            chunks.append("")

        caption = latex_escape(claim) if claim else ""
        result_title = frame_title_escape(heading)
        if fig.exists() and caption:
            chunks.append(f"\\resultframe{{{result_title}}}{{{rel_fig}}}{{%")
            chunks.append(f"    {caption}}}")
        elif fig.exists():
            chunks.append(f"\\figureframe{{{result_title}}}{{{rel_fig}}}")
        elif caption:
            chunks.append(f"\\claimframe{{{result_title}}}{{%")
            chunks.append(f"    {caption}}}")
        chunks.append("")

    chunks.append(closer_frame(f"{int(number):02d}"))
    chunks.append(r"\end{document}")
    chunks.append("")
    return "\n".join(chunks)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python _emit_deck.py <lesson_dir> [...]", file=sys.stderr)
        return 2
    for raw in sys.argv[1:]:
        lesson_dir = Path(raw)
        number, _ = lesson_heading(lesson_dir)
        out = lesson_dir / "slides" / f"lesson_{number}.tex"
        out.parent.mkdir(exist_ok=True)
        out.write_text(emit_tex(lesson_dir), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

