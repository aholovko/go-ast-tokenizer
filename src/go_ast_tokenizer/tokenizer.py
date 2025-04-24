"""
Wrapper for the Go AST tokenizer.
"""

import ctypes
from pathlib import Path
from typing import ClassVar, Optional

SPECIAL_TOKENS = [
    "<IDENT>",
    "<LIT_INT>",
    "<LIT_FLOAT>",
    "<LIT_IMAG>",
    "<LIT_CHAR>",
    "<LIT_STRING>",
    "<COMMENT>",
    "<ASSIGN_OP>",
    "<BINARY_OP>",
    "<IF>",
    "<ELSE>",
    "<SWITCH>",
    "<FUNC>",
    "<CASE>",
    "<LBRACE>",
    "<RBRACE>",
    "<LPAREN>",
    "<RPAREN>",
    "<COLON>",
]


class GoASTTokenizer:
    _instance: ClassVar[Optional["GoASTTokenizer"]] = None
    _lib: ClassVar[Optional[ctypes.CDLL]] = None

    def __new__(cls, lib_path: Optional[Path] = None) -> "GoASTTokenizer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_lib(lib_path)
        return cls._instance

    @classmethod
    def _initialize_lib(cls, lib_path: Optional[Path]) -> None:
        if cls._lib is not None:
            return

        if lib_path is None:
            lib_path = Path(__file__).parent / "tokenizer" / "_tokenizer.so"
        else:
            lib_path = Path(lib_path).absolute()

        try:
            cls._lib = ctypes.cdll.LoadLibrary(str(lib_path))
        except OSError as e:
            raise RuntimeError(f"Failed to load shared library '{lib_path}'") from e

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        if self._lib is None:
            raise RuntimeError("Library not initialized properly")

        # bind tokenize(src *C.char) -> *C.char
        self._tokenize = self._lib.tokenize
        self._tokenize.argtypes = [ctypes.c_char_p]
        self._tokenize.restype = ctypes.c_void_p

        # bind freeCString(ptr *C.char)
        self._free_cstr = self._lib.freeCString
        self._free_cstr.argtypes = [ctypes.c_void_p]
        self._free_cstr.restype = None

        self._initialized = True

    def tokenize(self, src: str) -> Optional[str]:
        if not src:
            raise ValueError("Input string to tokenize() must not be empty")

        ptr = self._tokenize(src.encode("utf-8"))
        if not ptr:
            raise RuntimeError("Go tokenizer returned NULL pointer")

        raw_str = ctypes.string_at(ptr)
        try:
            return raw_str.decode("utf-8")
        finally:
            self._free_cstr(ptr)
