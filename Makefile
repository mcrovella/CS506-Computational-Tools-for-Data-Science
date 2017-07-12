SRCS = \
01-Intro-to-Python.ipynb \
02A-Getting-Started.ipynb \
02B-Pandas.ipynb \
03-Probability-and-Statistics-Refresher.ipynb \
04-Linear-Algebra-Refresher.ipynb \
05-Distances-Timeseries.ipynb 

TEXS=$(SRCS:.ipynb=.tex)

TGTS=$(SRCS:.ipynb=.pdf)

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










