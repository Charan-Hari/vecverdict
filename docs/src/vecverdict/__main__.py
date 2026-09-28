"""Enable ``python -m vecverdict`` as an alias for the ``vecverdict`` script.

Console scripts are not always on PATH (unactivated virtualenvs, ``pipx run``,
CI images that install with ``--target``). Module execution always works.
"""

from __future__ import annotations

import sys

from vecverdict.cli import main

if __name__ == "__main__":
    sys.exit(main())
