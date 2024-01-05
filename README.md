# Getting started

### Installing Python 3.11.0
Use pyenv to install python 3.11.0: https://github.com/pyenv/pyenv

### Installing poetry
Poetry is used as a package manager: https://python-poetry.org/docs/#installation

### Setting up the repository and environment
To create the virtual environment with the Python packages run:

    make dev-setup

Source the vitual environment using:

    source .venv/bin/activate

### Running an experiment

NOTE: data is not uploaded due to IP restrictions.

Default experiment folder is `src/experiments`. Run the following commands to create experiments (perhaps with the `--overwrite` flag):

`python manage.py LookbackExperiment`
`python manage.py LoadShiftingExperiment`
`python manage.py mFRRExperiment`

Plots are created in the `src/plots/plot.py` and `src/plots/plot2.py`.