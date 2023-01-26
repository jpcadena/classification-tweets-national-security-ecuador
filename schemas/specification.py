"""
Specification schema
"""
from abc import ABC, abstractmethod
from datetime import datetime


class BaseSpecification(ABC):
    """
    Specification class
    """

    @abstractmethod
    def __init__(self, spec: str, lang: str = 'es') -> None:
        """
        Abstract constructor for Base object
        :param spec: Specification to be created
        :type spec: str
        :param lang: language to use, by default spanish
        :type lang: str
        """

    @abstractmethod
    def __or__(self, *args: tuple) -> None:
        """
        OR method to create OrSpecification
        :param args: Multiple Specifications
        :type args: tuple
        :return: OrSpecification instance
        :rtype: OrSpecification
        """


class Specification(BaseSpecification):
    """
    Specification class
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.spec: str = spec + ' lang:' + lang
        self.lang: str = 'lang:' + lang

    def __or__(self, *args: tuple):
        """
        OR method to create OrSpecification
        :param args: Multiple Specifications
        :type args: tuple
        :return: OrSpecification instance
        :rtype: OrSpecification
        """
        return OrSpecification(self, *args)


class OrSpecification(Specification):
    """
    OrSpecification class based on Specification
    """

    def __init__(self, lang: str = 'es', *args: tuple) -> None:
        self.args: str = ' OR '.join(map(str, list(args)))
        super().__init__(self.args, lang)


class TextSpecification(Specification):
    """
    Text Specification class based on Specification
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.text: str = spec + ' #ecuador'
        super().__init__(self.text, lang)


class SenderSpecification(Specification):
    """
    Sender Specification class based on Specification
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.sender: str = 'from:' + spec
        super().__init__(self.sender, lang)


class ReceiverSpecification(Specification):
    """
    Receiver Specification class based on Specification
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.receiver: str = 'to:' + spec
        super().__init__(self.receiver, lang)


class IncludeUserSpecification(Specification):
    """
    Include User Specification class based on Specification
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.user: str = '@' + spec
        super().__init__(self.user, lang)


class HashtagSpecification(Specification):
    """
    Hashtag Specification class based on Specification
    """

    def __init__(self, spec: str, lang: str = 'es') -> None:
        self.hashtag: str = '#' + spec + ' #ecuador'
        super().__init__(self.hashtag, lang)


class MultipleHashtagsSpecification(Specification):
    """
    Multiple Hashtags Specification class based on Specification
    """

    def __init__(self, lang: str = 'es', *args: tuple) -> None:
        self.multiple_hashtags: str = ' OR '.join(
            map(str, ['#' + arg for arg in list(args)]))
        super().__init__(self.multiple_hashtags, lang)


class DateSpecification(Specification):
    """
    Date Specification class based on Specification
    """

    def __init__(
            self, spec: str, lang: str = 'es', **kwargs: dict[str, datetime]
    ) -> None:
        text: str = spec
        if 'since' in kwargs:
            self.since: datetime = kwargs['since']
            text = text + 'since:' + self.since.strftime('YYYY-MM-DD')
        if 'until' in kwargs:
            self.until: datetime = kwargs['until']
            text = text + 'until:' + self.until.strftime('YYYY-MM-DD')
        self.text: str = text
        super().__init__(self.text, lang)
