
MachineLearning.pdf: MachineLearning.tex content.tex MachineLearning.bib
	xelatex -shell-escape MachineLearning.tex
	bibtex MachineLearning
	xelatex -shell-escape MachineLearning.tex

force: content.tex
	xelatex -shell-escape MachineLearning.tex
	bibtex MachineLearning
	xelatex -shell-escape MachineLearning.tex


content.tex: content.org extract_body.py code/*
# Export file to LaTeX
	emacs --script ./org-files-to-tex.el *.org
# Rename so that filename starts with underscore
	mv content.tex _content.tex
# Extract body and save result to the original filename
	python extract_body.py _content.tex content.tex
# Remove temporary file
	rm _content.tex

clean:
	rm -f MachineLearning.aux MachineLearning.bbl MachineLearning.blg MachineLearning.log MachineLearning.out MachineLearning.pdf
	rm -f _content.tex content.tex todo.tex
	rm -rf _minted-MachineLearning/

update-compiled-MachineLearning.pdf:
	make force
