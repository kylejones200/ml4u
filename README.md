# Machine Learning for Power & Utilities

A LaTeX book on applying AI and machine learning in the electric utility industry.

## Repository Structure

```
ml4u/
├── latex/          # LaTeX source files (main.tex + chapter files)
├── code/           # Python code examples referenced in chapters
├── images/         # Figures and diagrams
├── _internal/      # Bibliography files and internal documentation
└── scripts/        # Utility scripts for conversion and maintenance
```

## Building the Book

### Prerequisites

- LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
- `biber` for bibliography processing
- Python 3.8+ (for utility scripts)

### Build Instructions

1. Navigate to the `latex/` directory:
   ```bash
   cd latex
   ```

2. Build the PDF:
   ```bash
   make pdf
   ```

   Or manually:
   ```bash
   pdflatex main.tex
   biber main
   pdflatex main.tex
   pdflatex main.tex
   ```

3. The PDF will be generated in `book_output/main.pdf`

### Makefile Targets

- `make pdf` - Build the complete PDF with bibliography
- `make clean` - Remove auxiliary files (.aux, .log, .bbl, etc.)
- `make bib` - Process bibliography only
- `make help` - Show available targets

## Chapter Structure

The book is organized into parts:

- **Part I: Foundations** (Chapters 1-3)
- **Part II: Core Applications** (Chapters 4-9)
- **Part III: Advanced Techniques** (Chapters 10-13, 27-28)
- **Part IV: Integration and Scale** (Chapters 14-26)

## Code Examples

All Python code examples are in the `code/` directory, prefixed with chapter numbers (e.g., `c1_intro_to_ML.py`). Code is referenced in LaTeX using `\lstinputlisting` with line ranges.

## Images

All figures and diagrams are in the `images/` directory, prefixed with chapter numbers (e.g., `c1_chapter1_load_plot.png`).

## Bibliography

Bibliography files are in `_internal/`:
- `bibtex_library.bib` - Academic and technical references
- `case_studies.bib` - Industry case studies

## Maintenance

### Converting Markdown to LaTeX

If you need to regenerate LaTeX from markdown sources (if they exist):

```bash
python3 scripts/convert_md_to_latex.py
python3 scripts/clean_latex_files.py
```

### Cleaning LaTeX Files

Remove pandoc preamble and horizontal rules:

```bash
python3 scripts/clean_latex_files.py
```

## License

Copyright © 2025 Kyle Jones. All rights reserved.
