from fastapi import APIRouter, HTTPException
from app.models import QueryModel
from app.services.summarizer_service import display_summarized_results
from app.config import client, together_client

router = APIRouter()

@router.post("/summarize")
async def summarize(query: QueryModel):
    try:
        # Call the summarizer service function to get the summary
        summary = display_summarized_results(client, together_client, "rag_with_HF", query.query_text, k=4, lambda_mult=0.25)
        return {"summary": summary} if summary else {"message": "No relevant documents found."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
