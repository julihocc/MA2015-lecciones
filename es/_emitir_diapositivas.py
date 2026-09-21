"""Emite un deck Beamer al contrato de la Lección 01.

Los listados son \\lstinputlisting del script en disco (nunca código pegado).
Los pies de figura salen de la tabla del README de la lección.

Uso, desde la raíz del curso:

    python es/_emitir_diapositivas.py es/Leccion_01_Intro
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from _puentes import marco_cierre, marco_intro

BANDA_NUEVO = re.compile(r"^\s*# --- NUEVO \((\d+)\)\s+(.+?)\s+---")
CAMBIADO_CUALQUIERA = re.compile(r"# --- CAMBIADO ---")
CAMBIADO_BLOQUE = re.compile(r"^\s*# --- CAMBIADO ---")
CIERRE = re.compile(r"^\s*# -{3,}\s*$")
ITEM_RECETA = re.compile(r"^\s+(\d+)\.\s+(.+)$")
LINEA_TITULO = re.compile(r"^Lecci[oó]n\s+\d+\s+-\s+.+:\s+(.*)$")
SCRIPT_EN_CELDA = re.compile(r"`([^`]+\.py)`")
PREFIJO_PASO = re.compile(r"^(.+_\d{2})_")
ENCABEZADO_README = re.compile(
    r"#\s+Lecci[oó]n\s+(\d+)\s+[—–-]\s+(.*)$"
)
ENCABEZADOS_TABLA = {
    "Script",
    "Qué agrega",
    "Qué añade",
    "Qué demuestra su salida",
    "Qué prueba su salida",
    "Lo que agrega",
    "Lo que prueba su salida",
    "Resultado verificado",
}


def escape_latex(texto: str) -> str:
    texto = texto.replace("\\", r"\textbackslash{}")
    texto = texto.replace("&", r"\&")
    texto = texto.replace("%", r"\%")
    texto = texto.replace("#", r"\#")
    texto = texto.replace("_", r"\_")
    texto = texto.replace("{", r"\{")
    texto = texto.replace("}", r"\}")
    texto = texto.replace("~", r"\textasciitilde{}")
    texto = texto.replace("^", r"\textasciicircum{}")
    texto = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", texto)
    texto = re.sub(r"`([^`]+)`", r"\\texttt{\1}", texto)
    return texto


def escape_titulo(texto: str) -> str:
    return (
        texto.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def scripts_numerados(src: Path) -> list[Path]:
    return sorted(
        p for p in src.glob("*.py") if re.search(r"_\d{2}_", p.name)
    )


def docstring_de(texto: str) -> str:
    if '"""' not in texto:
        return ""
    return texto.split('"""', 2)[1]


def titulo_paso(doc: str, stem: str) -> str:
    for linea in doc.splitlines():
        m = LINEA_TITULO.match(linea.strip())
        if m:
            return m.group(1).strip()
    return stem.replace("_", " ")


def prosa_pregunta(doc: str) -> list[str]:
    lineas = doc.splitlines()
    i = 0
    while i < len(lineas):
        s = lineas[i].strip()
        if LINEA_TITULO.match(s) or s.startswith("===="):
            i += 1
            continue
        if s.startswith("NUEVO EN ESTE PASO"):
            i += 1
            while i < len(lineas) and lineas[i].strip() != "":
                i += 1
            continue
        if s == "":
            i += 1
            continue
        break
    cuerpo: list[str] = []
    while i < len(lineas):
        s = lineas[i].strip()
        if s.startswith("CAMBIOS RESPECTO A") or s.startswith("Ejecútalo:"):
            break
        if s.startswith("Ejecución:") or s.startswith("Run it:"):
            break
        if s.startswith("ELIMINADO") or s.startswith("REMOVED FROM"):
            break
        cuerpo.append(lineas[i].rstrip())
        i += 1
    while cuerpo and cuerpo[-1] == "":
        cuerpo.pop()
    parrafos: list[list[str]] = [[]]
    for linea in cuerpo:
        if linea.strip() == "":
            if parrafos[-1]:
                parrafos.append([])
            continue
        parrafos[-1].append(linea.strip())
    parrafos = [p for p in parrafos if p]
    return parrafos[0] if parrafos else []


def etiqueta_receta(resto: str) -> str:
    if "  " in resto:
        return resto.split("  ", 1)[0].strip()
    return resto.strip()


def items_receta(doc: str) -> list[tuple[int, str]]:
    items = []
    en_receta = False
    for linea in doc.splitlines():
        if "CAMBIOS RESPECTO A" in linea or "Introdúcelos en este orden" in linea:
            en_receta = True
            continue
        if en_receta and (
            linea.strip().startswith("Ejecútalo:")
            or linea.strip().startswith("Ejecución:")
            or linea.strip().startswith("Run it:")
        ):
            break
        if en_receta:
            m = ITEM_RECETA.match(linea)
            if m:
                items.append((int(m.group(1)), etiqueta_receta(m.group(2))))
    return items


def rangos_nuevo(texto: str) -> dict[int, tuple[int, int, str]]:
    lineas = texto.splitlines()
    hallados: dict[int, tuple[int, int, str]] = {}
    i = 0
    while i < len(lineas):
        m = BANDA_NUEVO.match(lineas[i])
        if not m:
            i += 1
            continue
        numero = int(m.group(1))
        nombre = m.group(2).strip()
        inicio = i
        fin = i
        j = i + 1
        while j < len(lineas):
            if BANDA_NUEVO.match(lineas[j]) or CAMBIADO_BLOQUE.match(lineas[j]):
                fin = j - 1
                break
            if CIERRE.match(lineas[j]):
                fin = j
                break
            j += 1
        else:
            fin = len(lineas) - 1
        hallados[numero] = (inicio + 1, fin + 1, nombre)
        i = fin + 1
    return hallados


def rangos_cambiado(texto: str) -> list[tuple[int, int]]:
    lineas = texto.splitlines()
    rangos: list[tuple[int, int]] = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        if not CAMBIADO_CUALQUIERA.search(linea):
            i += 1
            continue
        inicio = i
        if CAMBIADO_BLOQUE.match(linea) or (
            linea.lstrip().startswith("#") and CAMBIADO_CUALQUIERA.search(linea)
        ):
            j = i + 1
            vio_codigo = False
            while j < len(lineas):
                nxt = lineas[j]
                if BANDA_NUEVO.match(nxt) or CAMBIADO_BLOQUE.match(nxt):
                    break
                if CIERRE.match(nxt) and vio_codigo:
                    j += 1
                    break
                s = nxt.strip()
                if s.startswith("print(") and vio_codigo:
                    break
                if s and not s.startswith("#"):
                    vio_codigo = True
                j += 1
            fin = j - 1
        else:
            fin = i
        rangos.append((inicio + 1, fin + 1))
        i = fin + 1
    return rangos


def plan_listados(
    texto: str, receta: list[tuple[int, str]]
) -> list[tuple[str, int, int]]:
    nuevos = rangos_nuevo(texto)
    cambiados = rangos_cambiado(texto)
    i_cambiado = 0
    plan: list[tuple[str, int, int]] = []
    for numero, nombre in receta:
        if numero in nuevos:
            primero, ultimo, nom_banda = nuevos[numero]
            plan.append((f"({numero}) {nom_banda}", primero, ultimo))
            continue
        if i_cambiado < len(cambiados):
            primero, ultimo = cambiados[i_cambiado]
            i_cambiado += 1
            plan.append((f"({numero}) {nombre}", primero, ultimo))
    return plan


def afirmaciones_readme(readme: Path) -> dict[str, str]:
    afirmaciones: dict[str, str] = {}
    if not readme.exists():
        return afirmaciones
    for linea in readme.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|"):
            continue
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) < 2 or celdas[0] in {"#", "---"}:
            continue
        script = None
        for celda in celdas:
            m = SCRIPT_EN_CELDA.search(celda)
            if m:
                script = m.group(1)
        if script is None:
            continue
        afirmacion = None
        for celda in reversed(celdas):
            if SCRIPT_EN_CELDA.search(celda):
                continue
            if re.fullmatch(r"\d+", celda):
                continue
            if celda in ENCABEZADOS_TABLA:
                continue
            afirmacion = celda
            break
        if afirmacion:
            afirmaciones[script] = afirmacion
            clave = PREFIJO_PASO.match(script)
            if clave:
                afirmaciones.setdefault(clave.group(1), afirmacion)
    return afirmaciones


def encabezado_leccion(leccion_dir: Path) -> tuple[str, str]:
    readme = leccion_dir / "README.md"
    if readme.exists():
        primera = readme.read_text(encoding="utf-8").splitlines()[0].strip()
        m = ENCABEZADO_README.match(primera)
        if m:
            return f"{int(m.group(1)):02d}", m.group(2).strip()
    partes = leccion_dir.name.split("_")
    numero = partes[1]
    titulo = " ".join(partes[2:]).replace("_", " ")
    return numero, titulo


def afirmacion_para(nombre: str, afirmaciones: dict[str, str]) -> str:
    if nombre in afirmaciones:
        return afirmaciones[nombre]
    clave = PREFIJO_PASO.match(nombre)
    if clave and clave.group(1) in afirmaciones:
        return afirmaciones[clave.group(1)]
    if clave:
        nn = clave.group(1).rsplit("_", 1)[-1]
        archivos = [
            k
            for k in afirmaciones
            if k.endswith(".py") and f"_{nn}_" in k
        ]
        if len(archivos) == 1:
            return afirmaciones[archivos[0]]
    return ""


def carpeta_figuras(leccion_dir: Path) -> tuple[Path, str]:
    figuras = leccion_dir / "figuras"
    if figuras.exists():
        return figuras, "figuras"
    return leccion_dir / "figures", "figures"


def emitir_tex(leccion_dir: Path) -> str:
    src = leccion_dir / "src"
    fig_dir, fig_nombre = carpeta_figuras(leccion_dir)
    afirmaciones = afirmaciones_readme(leccion_dir / "README.md")
    numero, titulo = encabezado_leccion(leccion_dir)
    trozos: list[str] = [
        r"\input{../../../templates/es/preambulo.tex}",
        "",
        r"% Los listados se toman de src/ con \lstinputlisting; la diapositiva es el archivo.",
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
        r"\newcommand{\marcoresultado}[3]{%",
        r"    \begin{frame}{#1}",
        r"        \begin{center}",
        r"            \includegraphics[height=0.62\textheight,width=\textwidth,keepaspectratio]{#2}",
        r"        \end{center}",
        r"        \vspace{0.25em}",
        r"        {\small #3\par}",
        r"    \end{frame}%",
        r"}",
        "",
        r"\newcommand{\marcoafirmacion}[2]{%",
        r"    \begin{frame}{#1}",
        r"        {\small #2\par}",
        r"    \end{frame}%",
        r"}",
        "",
        r"\date{}",
        "",
        r"\begin{document}",
        "",
        f"\\tituloleccion{{{numero}}}{{{titulo}}}",
        "",
    ]
    trozos.append(marco_intro(f"{int(numero):02d}"))
    trozos.append("")

    for script in scripts_numerados(src):
        texto = script.read_text(encoding="utf-8")
        doc = docstring_de(texto)
        encabezado = titulo_paso(doc, script.stem)
        prosa = prosa_pregunta(doc)
        receta = items_receta(doc)
        listados = plan_listados(texto, receta)
        afirmacion = afirmacion_para(script.name, afirmaciones)
        fig = fig_dir / f"{script.stem}.png"
        rel_script = f"../src/{script.name}"
        rel_fig = f"../{fig_nombre}/{script.stem}.png"

        trozos.append(f"% ================= {script.stem} =================")
        trozos.append(f"\\section{{{escape_titulo(encabezado)}}}")
        trozos.append("")
        trozos.append(f"\\begin{{frame}}{{{escape_titulo(encabezado)}}}")
        if prosa:
            for linea in prosa:
                trozos.append(escape_latex(linea))
        else:
            trozos.append(escape_latex(encabezado))
        trozos.append(r"\end{frame}")
        trozos.append("")

        for titulo_crudo, primero, ultimo in listados:
            n_lineas = ultimo - primero + 1
            trozos.append(
                r"\begin{frame}[fragile]{" + escape_titulo(titulo_crudo) + "}"
            )
            estilo = r"basicstyle=\ttfamily\fontsize{8pt}{9.6pt}\selectfont"
            if n_lineas > 28:
                estilo = r"basicstyle=\ttfamily\fontsize{6pt}{7.2pt}\selectfont"
            elif n_lineas > 18:
                estilo = r"basicstyle=\ttfamily\fontsize{7pt}{8.5pt}\selectfont"
            trozos.append(
                f"\\lstinputlisting[{estilo},"
                f"firstline={primero},lastline={ultimo}]{{{rel_script}}}"
            )
            trozos.append(r"\end{frame}")
            trozos.append("")

        pie = escape_latex(afirmacion) if afirmacion else ""
        titulo_res = escape_titulo(encabezado)
        if fig.exists() and pie:
            trozos.append(f"\\marcoresultado{{{titulo_res}}}{{{rel_fig}}}{{%")
            trozos.append(f"    {pie}}}")
        elif fig.exists():
            trozos.append(f"\\marcfigura{{{titulo_res}}}{{{rel_fig}}}")
        elif pie:
            trozos.append(f"\\marcoafirmacion{{{titulo_res}}}{{%")
            trozos.append(f"    {pie}}}")
        trozos.append("")

    trozos.append(marco_cierre(f"{int(numero):02d}"))
    trozos.append(r"\end{document}")
    trozos.append("")
    return "\n".join(trozos)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Uso: python es/_emitir_diapositivas.py <carpeta_leccion> [...]",
            file=sys.stderr,
        )
        return 2
    for bruto in sys.argv[1:]:
        leccion_dir = Path(bruto)
        numero, _ = encabezado_leccion(leccion_dir)
        salida = leccion_dir / "slides" / f"leccion_{numero}.tex"
        salida.parent.mkdir(exist_ok=True)
        salida.write_text(emitir_tex(leccion_dir), encoding="utf-8")
        print(f"escribió {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

