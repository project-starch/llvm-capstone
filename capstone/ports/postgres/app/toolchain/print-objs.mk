# Prints a PGXS-style Makefile's objects (build-domain.sh: a module's objects): OBJS for a MODULE_big,
# one <name>.o per name for a MODULES list (contrib/test_decoding declares no OBJS).
print-objs:
	@echo $(OBJS) $(addsuffix .o,$(MODULES))
