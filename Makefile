DOCNAME = book

.PHONY: figures movefigures book

movefigures:
	(cd json; python generate-configs.py)
	scp json/Fig*json crovella@csa2.bu.edu:www/cs132-figures
	ssh crovella@csa2.bu.edu 'chmod a+r ~/www/cs132-figures/*'
	scp json/config*.json crovella@csa2.bu.edu:www/diagramar
	ssh crovella@csa2.bu.edu 'chmod a+r ~/www/diagramar/*'

book:
	jupyter-book build .
	cp -r _build/html/* docs
	git add docs
	git commit -am .
	git push
