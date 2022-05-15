# $`\tau`$ error definition

This file documents the contents of this directory: the source code to generate the $`\tau`$ error definition change.
[View the pretty rendered documentation here.](../tau_error_definition.md)

## LaTeX

The main figures in the rendered document are created using the LaTeX Tikz package.
There are two versions of the figure, with only slight requirements.
This is accomplished with a single `base.tex` file that uses `\ifnew` for conditionally
adding nodes that only appear in the new τ error definition.
The files `new.tex` and `old.tex` only set up this conditional, set its value, and include the base file.
Compiling `new.tex` will output `new.svg`. Similarly, compiling `old.tex` will output `old.svg`.

Normally, LaTeX outputs a PDF document. This cannot be included in markdown, so we convert it to an SVG.
This is accomplished by the options to the `standalone` document class:

```tex
\documentclass[crop,tikz,convert={outext=.svg,command=\unexpanded{pdf2svg \infile\space\outfile}},multi=false]{standalone}
```

**This requires that [pdf2svg](https://github.com/dawbarton/pdf2svg) be installed.**

The LaTeX engine I recommend is XeLaTex. Compile either document using the command:
```bash
latexmk -pdflatex=lualatex -shell-escape -pvc -interaction=nonstopmode -pdf -pvc {old,new}.tex
```

## Figures

The file `figures.py` is used to create the small matplotlib figures shown in some nodes of the main figures.
