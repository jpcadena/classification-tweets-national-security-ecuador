"""
Medium schema
"""
from typing import Optional
from pydantic import BaseModel

# TODO: clean Tweet structure to define this schema


class Medium(BaseModel):
    """
    Medium class based on Pydantic Base Model
    """


class Photo(Medium):
    """
    Photo class based on Medium
    """
    previewUrl: str
    fullUrl: str


class VideoVariant(BaseModel):
    """
    Video Variant class based on Pydantic Base Model
    """
    contentType: str
    url: str
    bitrate: Optional[int]


class Video(Medium):
    """
    Video class based on Medium
    """
    thumbnailUrl: str
    variants: list[VideoVariant]
    duration: Optional[float] = None
    views: Optional[int] = None


class Gif(Medium):
    """
    Gif class based on Medium
    """
    thumbnailUrl: str
    variants: list[VideoVariant]
