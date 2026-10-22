# controls-mapped.dom

`controls-mapped.c` is the Juliet runner's `capstone/bug-corpora/postgres/c-repros/shared/controls.c` plus one mode,
`uaf-mapped` (the diff is that mode and the usage strings). Built with the Juliet spatial census SDK's compiler, as
juliet/build.py builds controls.dom:

    <mnt-hdd>/capstone/juliet-sp/sdk/capstone-cc -O0 controls-mapped.c -o controls-mapped.dom   # in common.clean_env()

- capstone-cc sha256 7f3077ae894a9edb...
- the same command on the unmodified controls.c reproduces controls.dom b1decb0612856121 byte for byte (the census's)
- controls-mapped.dom sha256 fb30eab5a2fe4c20... (the image lives at <mnt-hdd>/capstone/uaf-mapped/; binaries are not
  committed)
