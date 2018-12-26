
base.pdf: base.tex content.tex base.bib
	xelatex -shell-escape base.tex
	bibtex base
	xelatex -shell-escape base.tex

force: content.tex
	xelatex -shell-escape base.tex
	bibtex base
	xelatex -shell-escape base.tex


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
	rm -f base.aux base.bbl base.blg base.log base.out base.pdf
	rm -f _content.tex content.tex todo.tex
	rm -rf _minted-base/

update-compiled-base.pdf:
	make force
