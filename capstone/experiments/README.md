# Experiments

What the paper measures on this platform, one directory per experiment, named after the
paper's plan. Deliberately outside `ports/`, which holds the ports and their build, and outside
`bug-corpora/`, which holds one program's defects: an experiment links its own instrument or
probes into a port's domain and records what it measured.

    a1-sqlite-reuse/    A1: does level 0 see the objects in the custom allocator? SQLite's
                        lookaside pool over memsys5, unprotected and under the Sublet port

Each directory has a `run.sh` that builds through the port, runs its control first and prints a
verdict against what the directory records; a control that fails exits 75 with no verdict, as in
the bug corpora. `results/<stamp>/` holds a pass as raw text -- the instrument's lines, the
program's own statistics, the images' sha256, a provenance record -- and nothing derived. The
figures are the paper's, drawn from the one stamp it pins and copies.
