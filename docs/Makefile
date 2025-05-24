.ONESHELL:

BLD_DIR = ./__build
SRC_DIR = ./__source

local_clean:
	rm -rf $(BLD_DIR)/*

local_html:
	@make local_clean
	sphinx-build -b html ${SRC_DIR} ${BLD_DIR}/html
ifdef VERSION_RELEASE
	cp -a $(BLD_DIR)/html/. $(BLD_DIR)/html/$(VERSION_RELEASE)
endif

local_pdf:
	@make local_clean
	sphinx-build -M simplepdf ${SRC_DIR} ${BLD_DIR}/pdf
