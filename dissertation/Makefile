all: build clean wordcount

build: build1 biber build2

build1:
	xelatex Dissertation.tex

biber:
	biber Dissertation

biberchapters:
	biber chapters/1-Introduction
	biber chapters/2-Preparation
	biber chapters/3-Implementation
	biber chapters/4-Evaluation
	biber chapters/5-Conclusions

build2:
	xelatex Dissertation.tex
	xelatex Dissertation.tex

clean:
	rm -f *.log *.aux *.out *.bbl *.bcf *.blg *.xml
cleanchapters:
	rm -f chapters/*.log chapters/*.aux chapters/*.out chapters/*.bbl chapters/*.bcf chapters/*.blg chapters/*.xml

wordcount:
	detex DissertationContent.tex | tr -cd '0-9A-Za-z \n' | wc -w

linecount:
	cloc --force-lang "Python",py ../src
