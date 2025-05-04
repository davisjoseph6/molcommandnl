# INCLUDE COMMON SETTINGS
include config/env

.PHONY: info

info:
	clear
	@printf "\ninfo:: Semantic interpreter approach\n\n"
	@printf "info:: common targets: init, populate, chat, chats, test\n"
	@printf "info::                 test-llmclient, test-interpreter\n"
	@/bin/ls -GF
	@printf "\n"
	@git status

populate:
	$(info )
	$(info info:: populating the vector database)
	$(info )
	$(info info:: odsl DSL samples)
	@ $(APYTHON) $(POPULATESCRIPT) --dsl odsl
	@ printf "\ninfo:: MolCommandNL DSL samples\n"
	@ $(APYTHON) $(POPULATESCRIPT) --dsl molcommand
	@ printf "\ninfo:: markdown DSL samples\n"
	@ $(APYTHON) $(POPULATESCRIPT) --dsl markdown

chats:
	$(info )
	$(info info:: chat interface, simple layout)
	@ $(APYTHON) $(CHATSCRIPT) --simple

chat:
	$(info )
	$(info info:: chat interface, advanced layout)
	@ $(APYTHON) $(CHATSCRIPT)

test:
	$(info )
	$(info info:: testing symantic interpreter)
	@ $(APYTHON) $(TESTSCRIPT)

test-llmclient:
	$(info )
	$(info info:: testing LLM client class)
	@ $(APYTHON) $(LLMCLIENT)

test-interpreter:
	$(info )
	$(info info:: testing LLM client class)
	@ $(APYTHON) $(SEMINTERP)

init:
	$(info )
	$(info info:: Initializing and populating the vector database)
	$(info )
	$(info info:: odsl DSL samples)
	@ $(APYTHON) $(POPULATESCRIPT) --dsl odsl --reset
	@ printf "\ninfo:: MolCommandNL DSL samples\n"
	@ $(APYTHON) $(POPULATESCRIPT) --dsl molcommand --reset
	@ printf "\ninfo:: markdown DSL samples\n"
	@ $(APYTHON) $(POPULATESCRIPT) --dsl markdown --reset

purge:
	$(info )
	$(info info:: cleaning common clutter)
	@rm -rf */__pycache__ $(CHROMAPATH)/odsl $(CHROMAPATH)/markdown
