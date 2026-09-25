# AGENTS.md

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

## Verification discipline

Named-target builds are for iteration, not for evidence. A target that is not
built is not a target that works, and `ctest -L <label>` runs only the binaries
that already exist.

- Before describing a branch, PR or change as healthy: run a plain
  `cmake --build` over all targets and a full `ctest`, and name the exclusions
  (`-E external-model,slow`) and the reason for each. A named-target build does
  not establish this.
- After changing shared code — options, fault-injection paths, link libraries,
  or anything a default build consumes — re-run the **default** path, not only
  the new mode. A change can leave every new test green while breaking the
  path that was verified earlier.
- Adding a source file used by a widely-consumed translation unit means
  auditing every target that compiles that unit. Test support libraries are
  consumers too, and they are not in the default build of the main binary.
- Do not call a check unrunnable before trying the container route. `nix` works
  under `podman` with the repo bind-mounted, which is how the pinned
  clang-format gate is run here.
- In any check you write, keep the expected value and the observed value
  independent. A prediction that is also its own result cannot fail.
- Prefer a pre-existing `main` checkout as the baseline when a check fails, so
  a new defect is not attributed to inherited state or vice versa.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
