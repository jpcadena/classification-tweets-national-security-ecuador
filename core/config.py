"""
Config script
"""
import os

from dotenv import load_dotenv
from numpy import uint8, uint16

dotenv_path: str = ".env"
load_dotenv(dotenv_path=dotenv_path)

MAX_COLUMNS: int = int(os.getenv("MAX_COLUMNS"))
WIDTH: int = int(os.getenv("WIDTH"))
CHUNK_SIZE: uint16 = uint16(os.getenv("CHUNK_SIZE"))
PALETTE: str = os.getenv("PALETTE")
FONT_SIZE: uint8 = uint8(os.getenv("FONT_SIZE"))
ENCODING: str = os.getenv("ENCODING")
FIG_SIZE: tuple[uint8, uint8] = (15, 8)
COLORS: list[str] = ["lightskyblue", "coral", "palegreen"]
NUMERICS: list[str] = [
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32",
    "int64",
    "float16", "float32", "float64"]
RANGES: list[tuple] = [
    (0, 255), (0, 65535), (0, 4294967295), (0, 18446744073709551615),
    (-128, 127), (-32768, 32767), (-2147483648, 2147483647),
    (-18446744073709551616, 18446744073709551615)]
PROJECT_NAME: str = os.getenv("PROJECT_NAME")
DATE_FORMAT: str = os.getenv("DATE_FORMAT")
