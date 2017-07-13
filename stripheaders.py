import sys

outputLines = ['\section{', '\subsection{', '\subsubsection{', '\paragraph']
for line in sys.stdin:
    # correct for missing package that is used in typesetting tables
    printIt = False
    for pattern in outputLines:
        if (line.find(pattern) > -1):
            printIt = True
    if printIt:
        print(line)
