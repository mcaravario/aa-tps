BINS = informe.pdf informe.out informe.aux informe.log

.PHONY: all clean

informe.pdf: informe.tex
		pdflatex $<
		xdg-open informe.pdf
all: $(BINS)

clean:
	rm -f $(BINS)
