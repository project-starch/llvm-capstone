# Smoke specimen: proves the VM executes bytecode and the answer reaches the host.
#
# It ADDS rather than returning a constant, for the same reason the WAMR test
# module does: a constant would pass even if the interpreter never ran an
# instruction, and this project has paid for that class of test more than once.
40 + 2
