from fastapi import APIRouter, HTTPException
from models import QueryModel
from services.summarizer_service import display_summarized_results
from config import together_client, S3_BUCKET_NAME, S3_CHROMADB_PATH

router = APIRouter()

@router.post("/summarize")
async def summarize(query: QueryModel):
    try:
        # Call the summarizer service function to get the summary
        summary = display_summarized_results(
            s3_bucket=S3_BUCKET_NAME,
            prefix=S3_CHROMADB_PATH,
            tg_client=together_client,
            query_text=query.query_text,
            k=4,
            lambda_mult=0.25
        )
        return {"summary": summary} if summary else {"message": "No relevant documents found."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
