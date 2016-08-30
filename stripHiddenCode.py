import sys

inVerbatim = 0
for line in sys.stdin:
    # correct for missing package that is used in typesetting tables
    if (line.find("usepackage{longtable}") > -1):
        print(r"\usepackage{booktabs}")
    # correct for the bug in nbconvert that is naming jpg files using .jpe
    if (line.find("maketitle") > -1):
        print(r"\DeclareGraphicsRule{.jpe}{jpg}{*}{}")
    if ((line.find("\\begin{Verbatim}") > -1) or (line.find("\\begin{verbatim}") > -1)):
        inVerbatim = 1
        vblock = line
    elif inVerbatim:
        vblock = vblock + line
        if ((line.find('\\end{Verbatim}') > -1) or (line.find('\\end{verbatim}') > -1)):
            inVerbatim = 0
            # Things that we omit as verbatim: blocks containing
            # hide_code_in_slideshow
            # %matplotlib 
            # IPython.core.display.HTML
            # %%html
            if ((vblock.find('hide\\PYZus{}code\\PYZus{}in\\PYZus{}slideshow') < 0) 
            and (vblock.find('\\PY{k}{matplotlib}') < 0)
            and (vblock.find('IPython.core.display.HTML') < 0)
            and (vblock.find('\\PY{o}{\\PYZpc{}\\PYZpc{}}\\PY{k}{html}') < 0)):
                print(vblock, end=" ")
    else:
        print (line, end=" ")
