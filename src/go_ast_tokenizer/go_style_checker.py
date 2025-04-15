"""
GoStyleChecker is a wrapper for the Go style checker shared library.
"""

import ctypes
from pathlib import Path
from typing import ClassVar, List, Optional


class GoStyleChecker:
    # fmt: off
    STYLE_WARNING_FLAGS: ClassVar[dict[int, str]] = {
        1 << 1: "assignOp",             # 2
        1 << 2: "builtinShadow",        # 4
        1 << 3: "captLocal",            # 8
        1 << 4: "commentFormatting",    # 16
        1 << 5: "defaultCaseOrder",     # 32
        1 << 6: "elseif",               # 64
        1 << 7: "ifElseChain",          # 128
        1 << 8: "importShadow",         # 256
        1 << 9: "newDeref",             # 512
        1 << 10: "paramTypeCombine",    # 1024
        1 << 11: "regexpMust",          # 2048
        1 << 12: "singleCaseSwitch",    # 4096
        1 << 13: "switchTrue",          # 8192
        1 << 14: "typeSwitchVar",       # 16384
        1 << 15: "typeUnparen",         # 32768
        1 << 16: "underef",             # 65536
        1 << 17: "unlambda",            # 131072
        1 << 18: "unslice",             # 262144
        1 << 19: "valSwap",             # 524288
        1 << 20: "wrapperFunc",         # 1048576
    }
    # fmt: on

    _instance: ClassVar[Optional["GoStyleChecker"]] = None
    _lib: ClassVar[Optional[ctypes.CDLL]] = None

    def __new__(cls, lib_path: Optional[Path] = None) -> "GoStyleChecker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_lib(lib_path)
        return cls._instance

    @classmethod
    def _initialize_lib(cls, lib_path: Optional[Path] = None) -> None:
        if cls._lib is not None:
            return

        if lib_path is None:
            lib_path = Path(__file__).parent / "checker" / "_checkstyle.so"
        else:
            lib_path = Path(lib_path).absolute()

        try:
            cls._lib = ctypes.cdll.LoadLibrary(str(lib_path))
        except OSError as exc:
            raise RuntimeError(f"Failed to load shared library '{lib_path}'") from exc

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        if self._lib is None:
            raise RuntimeError("Library not initialized properly")

        self.check_func = self._lib.check
        self.check_func.argtypes = [ctypes.c_char_p]
        self.check_func.restype = ctypes.c_int

        self._initialized = True

    def check_style(self, snippet: str) -> List[str]:
        mask = self.check_func(snippet.encode("utf-8"))
        if mask < 0:
            raise ValueError("Failed to check style of snippet")

        return [name for bit, name in self.STYLE_WARNING_FLAGS.items() if mask & bit]
