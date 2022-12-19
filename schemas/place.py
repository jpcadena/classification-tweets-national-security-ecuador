"""
Places schema
"""
from pydantic import BaseModel

# TODO: clean Tweet structure to define this schema


class Place(BaseModel):
    """
    Place class based on Pydantic Base Model
    """
    fullName: str
    name: str
    type: str
    country: str
    countryCode: str
