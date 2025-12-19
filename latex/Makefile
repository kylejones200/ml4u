# Makefile for building LaTeX book

MAIN = main
BIBFILE = ../_internal/case_studies.bib

# LaTeX compiler
LATEX = pdflatex
BIBER = biber
MAKEINDEX = makeindex

# Output directory
OUTPUT_DIR = ../book_output

.PHONY: all clean pdf bib index help

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIBFILE)
	@echo "Running pdflatex (first pass)..."
	$(LATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	@echo "Running biber for bibliography..."
	cd $(OUTPUT_DIR) && $(BIBER) $(MAIN)
	@echo "Running pdflatex (second pass)..."
	$(LATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	@echo "Running pdflatex (third pass for references)..."
	$(LATEX) -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	@echo "✓ PDF created: $(OUTPUT_DIR)/$(MAIN).pdf"

bib: $(BIBFILE)
	@echo "Processing bibliography..."
	cd $(OUTPUT_DIR) && $(BIBER) $(MAIN)

index:
	@echo "Creating index..."
	cd $(OUTPUT_DIR) && $(MAKEINDEX) $(MAIN).idx

clean:
	@echo "Cleaning LaTeX auxiliary files..."
	rm -f $(OUTPUT_DIR)/*.aux
	rm -f $(OUTPUT_DIR)/*.bbl
	rm -f $(OUTPUT_DIR)/*.bcf
	rm -f $(OUTPUT_DIR)/*.blg
	rm -f $(OUTPUT_DIR)/*.idx
	rm -f $(OUTPUT_DIR)/*.ilg
	rm -f $(OUTPUT_DIR)/*.ind
	rm -f $(OUTPUT_DIR)/*.lof
	rm -f $(OUTPUT_DIR)/*.log
	rm -f $(OUTPUT_DIR)/*.lot
	rm -f $(OUTPUT_DIR)/*.out
	rm -f $(OUTPUT_DIR)/*.run.xml
	rm -f $(OUTPUT_DIR)/*.toc
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  all     - Build PDF (default)"
	@echo "  pdf     - Build PDF with bibliography"
	@echo "  bib     - Process bibliography only"
	@echo "  index   - Create index"
	@echo "  clean   - Remove auxiliary files"
	@echo "  help    - Show this help"

