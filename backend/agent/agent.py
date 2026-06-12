from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from typing import Literal, Optional
from typing_extensions import Annotated, TypedDict
from langchain.messages import AnyMessage
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    contact_id: str


def risk_flag()