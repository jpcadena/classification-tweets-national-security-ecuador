"""
User schema
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, HttpUrl
from schemas.text_link import TextLink

# TODO: clean Tweet structure to define this schema


class User(BaseModel):
    """
    User class based on Pydantic Base Model
    """
    username: str
    id: int
    displayname: Optional[str] = None
    description: Optional[str] = None
    rawDescription: Optional[str] = None
    renderedDescription: Optional[str] = None
    descriptionLinks: Optional[list[TextLink]] = None
    verified: Optional[bool] = None
    created: Optional[datetime] = None
    followersCount: Optional[int] = None
    friendsCount: Optional[int] = None
    statusesCount: Optional[int] = None
    favouritesCount: Optional[int] = None
    listedCount: Optional[int] = None
    mediaCount: Optional[int] = None
    location: Optional[str] = None
    protected: Optional[bool] = None
    link: Optional[TextLink] = None
    profileImageUrl: Optional[str] = None
    profileBannerUrl: Optional[str] = None
    label: Optional['UserLabel'] = None


class UserLabel(BaseModel):
    """
    User Label class based on Pydantic Base Model
    """
    description: str
    url: Optional[HttpUrl] = None
    badgeUrl: Optional[str] = None
    longDescription: Optional[str] = None
