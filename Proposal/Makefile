all: build clean

build: build1 biber build2

biber: 
	biber DraftProposal
build1:
	xelatex DraftProposal.tex
build2:
	xelatex DraftProposal.tex
	xelatex DraftProposal.tex

clean: 
	rm *.log *.aux *.out *.bbl *.bcf *.blg *.xml
