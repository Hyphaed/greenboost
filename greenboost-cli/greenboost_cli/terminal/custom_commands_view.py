from pathlib import Path

from greenboost_cli import custom_commands as CC


def rows():
    out = []
    for name, (path, desc) in sorted(CC.discover().items()):
        src = "built in" if CC._SHIPPED in path.parents else str(path.parent).replace(str(Path.home()), "~")
        out.append((name, desc, src))
    return out
