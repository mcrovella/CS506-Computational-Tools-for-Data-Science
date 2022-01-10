IMAGE:=mcrovella/cs506-lectures
TAG?=latest
CONTNAME:=cs506

# force no caching for docker builds
#DCACHING=--no-cache

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
requirements.txt: ## compile requirements.txt from requirements.in
	pip-compile

help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: DARGS?=
build: INAME=$(IMAGE)
build: ## Make the container image
	docker build $(DARGS) $(DCACHING) --rm --force-rm -t $(INAME):$(TAG) .

# container will be removed on exit/stop -- ephemeral
run: ARGS?=
run: INAME=$(IMAGE)
run: PORT?=8888
run: ## create & run a jupyter notebook server container instance - ephemeral
	docker run -it --rm --name $(CONTNAME) -p $(PORT):8888 $(INAME):$(TAG) $(ARGS)

# these can be managed via: "docker stop cs506", "docker start cs506"
# and contents will be persistent
runkeep: ARGS?=
runkeep: INAME=$(IMAGE)
runkeep: PORT?=8888
runkeep: ## create a jupyter notebook server container instance - persistent
	docker run -it --name $(CONTNAME) -p $(PORT):8888 $(INAME):$(TAG) $(ARGS) 

push: DARGS?=
push: INAME?=$(IMAGE)
push: ## push container image to dockerhub
	docker push $(INAME):$(TAG)
