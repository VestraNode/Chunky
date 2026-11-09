# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

import importlib.util
import sys
from pathlib import Path


def _alias_package(alias: str, folder: str) -> None:
    path = Path(__file__).resolve().parent / folder
    name = f"{__name__}.{alias}"
    if name in sys.modules or not (path / "__init__.py").is_file():
        return
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                  submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    globals()[alias] = module


_alias_package("chunky11", "ChunkyV1.1")
