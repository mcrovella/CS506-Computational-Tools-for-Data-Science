
.PHONY: book

book:
        # -v verbose
        # -all re-build all pages not just changed pages
        # -W make warning treated as errors
        # -n nitpick generate warnings for all missing links
        # --keep-going despite -W don't stop delay errors till the end
	jupyter-book build -v -n --keep-going .

pushbook:
	cp -r _build/html/* docs
	git add docs
	git commit -m 'book update'
	git push

requirements.txt: requirements.in
	pip-compile
