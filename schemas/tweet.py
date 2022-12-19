"""
Tweet schema
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, HttpUrl

# TODO: clean Tweet structure to define this schema


class Tweet(BaseModel):
    """
    Tweet class based on Pydantic Base Model
    """
    url: HttpUrl
    date: datetime
    content: str
    rendered_content: str
    id: int
    user: 'User'
    reply_count: int
    retweet_count: int
    like_count: int
    quote_count: int
    conversation_id: int
    lang: str
    sourceLabel: Optional[HttpUrl] = None
    outlinks: Optional[list[str]] = None
    tcooutlinks: Optional[list[str]] = None
    media: Optional[list['Medium']] = None
    retweetedTweet: Optional['Tweet'] = None
    quotedTweet: Optional['Tweet'] = None
    inReplyToTweetId: Optional[int] = None
    inReplyToUser: Optional['User'] = None
    mentionedUsers: Optional[list['User']] = None
    coordinates: Optional['Coordinates'] = None
    place: Optional['Place'] = None
    hashtags: Optional[list[str]] = None
