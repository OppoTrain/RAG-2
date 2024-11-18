from pydantic import BaseModel

class QueryModel(BaseModel):
    query_text: str
