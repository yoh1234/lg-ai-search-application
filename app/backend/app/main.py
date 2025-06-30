from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from .chat_service import ChatService
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = FastAPI(title="LG Product Search Chatbot")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_service = ChatService(OPENAI_API_KEY)

# WebSocket 연결 관리
connections = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    connections[session_id] = websocket
    
    try:
        while True:
            # 사용자 메시지 받기
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                user_query = message.get("query", "")
                
                # thinking process 시뮬레이션
                await websocket.send_text(json.dumps({
                    "type": "thinking",
                    "step": "analyzing",
                    "description": "Analyzing your query..."
                }))
                
                await websocket.send_text(json.dumps({
                    "type": "thinking", 
                    "step": "searching",
                    "description": "Searching for products..."
                }))
                
                # 실제 검색 수행
                result = await chat_service.chat(session_id, user_query)
                
                # 결과 전송
                await websocket.send_text(json.dumps({
                    "type": "result",
                    "products": [p.dict() for p in result.products],
                    "explanation": result.explanation,
                    "total_count": result.total_count
                }))
                
    except WebSocketDisconnect:
        del connections[session_id]

@app.get("/")
async def root():
    return {"message": "LG Product Search Chatbot API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}