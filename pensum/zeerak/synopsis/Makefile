# Filename without extension
filename = synopsis

pdf-clean:
	make pdf
	make clean

pdf:
	pdflatex --shell-escape ${filename}
	bibtex ${filename}||true 
	pdflatex --shell-escape ${filename}
	pdflatex --shell-escape ${filename}
	open -a Preview ${filename}.pdf

clean:
	rm -f ${filename}.{log,aux,out,bbl,blg,pyg,toc}
	rm -f sections/*.aux

cleanup:
	rm -f ${filename}.{pdf,log,aux,out,bbl,blg,pyg,toc}
	rm -f sections/*.aux

tarball:
	mv main.pdf ../waseem_zeerak/waseem_zeerak.pdf
	tar -czvf waseem_zeerak.tar.gz waseem_zeerak
