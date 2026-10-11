# Historical motion prototypes

This package keeps the former motion models, Gaussian implementation, and
module/trainer source for regression tests and reading old results. The active
training path imports `pipeline/ReCon-GS/model/` and the top-level
`module.py`/`trainer.py` only. Historical training entry points are retired;
new runs use `main.py` with a rigid-tree configuration.
