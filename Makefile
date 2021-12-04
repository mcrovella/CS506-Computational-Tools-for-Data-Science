IMAGE:=mcrovella/cs506-lectures
TAG?=latest

.PHONY: book help

book:
book: ## compile the book but do not publish it
        # -v verbose
        # -all re-build all pages not just changed pages
        # -W make warning treated as errors
        # -n nitpick generate warnings for all missing links
        # --keep-going despite -W don't stop delay errors till the end
	jupyter-book build -v -n --keep-going .

pushbook:
pushbook: ## publish the last compiled book
	cp -r _build/html/* docs
	git add docs
	git commit -m 'book update'
	git push

requirements.txt: requirements.in
	pip-compile

help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


