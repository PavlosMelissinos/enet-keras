# Configuration files

This is the directory where the configuration for the experiments will be loaded from.

In order to run a custom experiment, copy the `solver.json.default` file into any other name, e.g. `train_enet.json` and run `python src/run.py ${mode} train_enet.json` from the root directory of the project. `mode` can only take the value of `train` at the moment. `predict` and `test`/`evaluate` functionality is planned, as well as a `python run.py --help` menu.
