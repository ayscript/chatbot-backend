from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# FastAPI app
app = FastAPI(title="Football Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ChatRequest(BaseModel):
    userText: str


def ai_response(userText: str) -> str:
    # Conversation history
    messageList = [
        {
            "role": "system",
            "content": (
                "You are a helpful football general knowledge expert. "
                "Provide accurate answers about football and its history only. "
                "Do NOT answer non-football questions."
            )
        }
    ]
    # Add user message
    messageList.append({
        "role": "user",
        "content": userText
    })

    # Create completion
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messageList,
        max_completion_tokens=500
    )

    answer = response.choices[0].message.content

    # Append assistant message
    messageList.append({
        "role": "assistant",
        "content": answer
    })

    return answer


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = ai_response(req.userText)
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )