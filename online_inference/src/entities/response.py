from pydantic import BaseModel


class Response(BaseModel):
    target: int
