DOCNAME = book

SRCS = \
L0Introduction.ipynb \
L0Clickers.ipynb \
L1LinearEquations.ipynb \
L2RowReductions.ipynb \
L3VectorEquations.ipynb

# TEXS=$(SRCS:.ipynb=.tex)

TGTS=$(SRCS:.ipynb=.pdf)

PUBLISHDIR = /cs-pub/www-dir/faculty/crovella/restricted/pebook

############################## shouldn't need to change below this line

LATEX  = pdflatex

.SUFFIXES: .ipynb .tex .pdf

%.tex: %.ipynb
	/bin/rm -rf tmpFile
	jupyter nbconvert $< --to latex
	# this is fixing a bug in ipython nbconvert 3.0 - misnames graphics files
	# sed 's/.jpe}/.jpeg}/g' < $@ > tmpFile
	mv $@ tmpFile
	python stripHiddenCode.py < tmpFile > $@
	rm tmpFile

%.pdf: %.tex
	$(LATEX) $<
	rm $*.out $*.log $*.aux

topleveltarget: $(TGTS)










