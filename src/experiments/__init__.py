import glob
import importlib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


modules = glob.glob(os.path.dirname(__file__) + "/*.py")
for module in modules:
    if module.endswith("__init__.py"):
        continue
    module = os.path.basename(module)[:-3]

    module_load = f"{__name__}.{module}"
    logger.info(f"Importing {module_load}")
    importlib.import_module(module_load)

# iterate over all folders in this directory
for folder in glob.glob(os.path.dirname(__file__) + "/*"):
    # iterate over all modules in this folder
    for module in glob.glob(folder + "/*.py"):

        module_load = (
            f"{__name__}."
            + os.path.basename(folder)
            + "."
            + os.path.basename(module)[:-3]
        )
        logger.info(f"Importing {module_load}")
        importlib.import_module(module_load)
