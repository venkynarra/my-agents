from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QueryRequest(_message.Message):
    __slots__ = ("query", "history")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    query: str
    history: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, query: _Optional[str] = ..., history: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class ContactFormRequest(_message.Message):
    __slots__ = ("name", "email", "message")
    NAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    email: str
    message: str
    def __init__(self, name: _Optional[str] = ..., email: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class ContactFormResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class MeetingRequest(_message.Message):
    __slots__ = ("email", "time", "message")
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    email: str
    time: str
    message: str
    def __init__(self, email: _Optional[str] = ..., time: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class MeetingResponse(_message.Message):
    __slots__ = ("success", "message", "event_link")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EVENT_LINK_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    event_link: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., event_link: _Optional[str] = ...) -> None: ...
