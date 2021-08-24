
.PHONY: book

book:
	jupyter-book build .
	cp -r _build/html/* docs
	git add docs
	git commit -m 'book update'
	git push
