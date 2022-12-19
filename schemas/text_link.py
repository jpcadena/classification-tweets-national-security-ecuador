"""
Text Link schema
"""
from typing import Optional
from pydantic import BaseModel, HttpUrl

# TODO: clean Tweet structure to define this schema


class TextLink(BaseModel):
    """
    Text Link class based on Pydantic Base Model
    """
    text: Optional[str]
    url: HttpUrl
    tcourl: str
    indices: tuple[int, int]
