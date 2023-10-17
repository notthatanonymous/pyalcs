install_deps:
	pip install -r requirements.txt --ignore-installed six
test:
	pytest
coverage:
	coverage run -m unittest discover -v tests
	coverage report -m
	rm .coverage
pep8:
	find . -name \*.py -exec pep8 --ignore=E129,E222,E402 {} +
notebook:
	jupyter notebook
