all: last_output.txt

last_output.txt: *ipynb
	mkdir -p nbconvert/html
	jupyter-nbconvert *ipynb --to python --output-dir=nbconvert/python
#	mkdir -p nbconvert/python
#	jupyter-nbconvert "$?" --to html --output-dir=nbconvert/html
	date > last_output.txt
	git rev-parse HEAD >> last_output.txt

clean:
	rm -rf nbconvert
	rm -f last_output.txt
