
.PHONY: book

book:
	jupyter-book build .
	cp -r _build/html/* docs
	git add docs
	git commit -am .
	git push
