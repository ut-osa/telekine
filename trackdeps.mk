DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

COMPILE.c = $(CC) $(DEPFLAGS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.hip = $(HIPCC) $(DEPFLAGS) $(HIPFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

%.o : %.c
%.o : %.c $(DEPDIR)/%.d
	@echo "    CC $@"
	@$(COMPILE.c) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

%.hip.o : %.cpp
%.hip.o : %.cpp $(DEPDIR)/%.d
	@echo "   HIPCC $@"
	@$(COMPILE.hip) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	@echo "   CXX $@"
	@$(COMPILE.cc) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(DEPDIR)/*.d)
