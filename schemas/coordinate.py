"""
Coordinates schema
"""
from pydantic import BaseModel

# TODO: clean Tweet structure to define this schema


class Coordinates(BaseModel):
    """
    Coordinates class based on Pydantic Base Model
    """
    longitude: float
    latitude: float
