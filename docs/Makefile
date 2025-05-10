.ONESHELL:

BLD_DIR = ./__build
SRC_DIR = ./__source

github_clean:
	find . \( -type d -maxdepth 1 ! -name "." \) \
	-and \( -type d -maxdepth 1 ! -name "*.*.*" \) \
	-and \( -type d -maxdepth 1 ! -name "_doc_*" \) \
	-exec rm -rf {} +
	rm -f *.html *.js *.inv .buildinfo

github_html:
	@make github_clean
	@make local_html
	cp -a $(BLD_DIR)/html/. .
ifdef VERSION_RELEASE
	cp -a $(BLD_DIR)/html/. ./$(VERSION_RELEASE)
endif

local_clean:
	rm -rf $(BLD_DIR)/*

local_html:
	@make local_clean
	sphinx-build -b html ${SRC_DIR} ${BLD_DIR}/html
