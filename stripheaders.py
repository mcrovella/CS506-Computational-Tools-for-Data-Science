import sys

outputLines = ['\section{', '\subsection{', '\subsubsection{', '\paragraph']
printIt = False
for line in sys.stdin:
    # correct for missing package that is used in typesetting tables
    for pattern in outputLines:
        if (line.find(pattern) > -1):
            printIt = True
    if printIt:
        print(line, end='')
    if (line.find('}') > -1):
        printIt = False
