.PHONY: build install docs

build:
	poetry build
    
install:
	poetry install && \
	poetry run pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu116/repo.html

docs:
	poetry install -E docs && poetry run pip install dgl==1.1.3