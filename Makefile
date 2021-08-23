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
	github checkout gh-pages
	github checkout 
	#ssh crovella@csa2.bu.edu '/bin/rm -rf ~/www/cs132-book'
	#scp -r _build/html crovella@csa2.bu.edu:www/cs132-book
	#ssh crovella@csa2.bu.edu 'chmod -R a+rx ~/www/cs132-book'







