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

# test-llmclient:
# 	$(info )
# 	$(info info:: testing LLM client class)
# 	@ $(APYTHON) $(LLMCLIENT)

# test-interpreter:
# 	$(info )
# 	$(info info:: testing LLM client class)
# 	@ $(APYTHON) $(SEMINTERP)

# Mapping of keywords to their corresponding scripts
TEST_SCRIPTS = \
	llmclient=$(LLMCLIENT) \
	interpreter=$(SEMINTERP) \
	tokenizer=scripts/test_tokenizer.py \
	parser=scripts/test_parser.py \
	renderer=scripts/test_renderer.py \
	# Add more as needed

# Extract script path for a given test
get-script = $(word 2, $(filter $1=%, $(TEST_SCRIPTS)))

# Pattern rule for all test-<keyword> targets
test-%:
	$(info )
	$(info info:: testing $*)
	echo $(APYTHON) $(call get-script,$*)

# Convenience alias to show available tests
.PHONY: list-tests
list-tests:
	@echo "Available tests:"
	@echo "$(TEST_SCRIPTS)" | tr ' ' '\n' | cut -d= -f1


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
