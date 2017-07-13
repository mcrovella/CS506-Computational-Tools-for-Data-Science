SRCS = \
01-Intro-to-Python.ipynb \
02A-Getting-Started.ipynb \
02B-Pandas.ipynb \
03-Probability-and-Statistics-Refresher.ipynb \
04-Linear-Algebra-Refresher.ipynb \
05-Distances-Timeseries.ipynb \
06A-Clustering-Overview.ipynb \
07-Clustering-II.ipynb \
08B-Clustering-III.ipynb \
09-Clustering-IV-GMM-EM.ipynb \
10-Low-Rank-and-SVD.ipynb \
11-Dimensionality-Reduction-SVD-II.ipynb \
12-Anomaly-Detection-SVD-III.ipynb \
15B-Classification-II-Demo.ipynb \
16B-Classification-III-Random-Projection.ipynb \
16C-Classification-III-SVM-Demo.ipynb \
17-Regression-I-Linear.ipynb \
18-Regression-II-Logistic.ipynb \
19-Regression-III-More-Linear.ipynb \
21B-MapReduce-Demo.ipynb \
22B-Networks-I-Intro-Demo.ipynb \
23-Networks-II-Centrality.ipynb \
24-Networks-III-Clustering.ipynb 

TEXS=$(SRCS:.ipynb=.tex)

TGTS=$(SRCS:.ipynb=.pdf)

HDRS=$(TEXS:.tex=.hdrs)

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

# rules for creating pdf are chains, which means intermediate (.tex) file
# would be automatically removed without the following
.PRECIOUS: %.tex

%.pdf: %.tex
	$(LATEX) $<
	rm $*.out $*.log $*.aux

%.hdrs: %.tex
	python stripheaders.py < $< > $@

topleveltarget: $(TEXS)
	echo $(TEXS)

toc.pdf: $(HDRS)
	cat preamble.tex $(HDRS) postamble.tex > toc.tex
	$(LATEX) toc.tex











