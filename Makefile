.PHONY: format lint all

format:
	isort jimg_int
	black jimg_int
	isort tests
	black tests

lint:
	pylint --exit-zero --disable=import-error,no-member jimg_int

	

all: format lint
