
DeepLearning.pdf: DeepLearning.tex content.tex DeepLearning.bib
	xelatex -shell-escape DeepLearning.tex
	bibtex DeepLearning
	xelatex -shell-escape DeepLearning.tex

force: content.tex
	xelatex -shell-escape DeepLearning.tex
	bibtex DeepLearning
	xelatex -shell-escape DeepLearning.tex


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
	rm -f DeepLearning.aux DeepLearning.bbl DeepLearning.blg DeepLearning.log DeepLearning.out DeepLearning.pdf
	rm -f _content.tex content.tex todo.tex
	rm -rf _minted-DeepLearning/

update-compiled-DeepLearning.pdf:
	make force
