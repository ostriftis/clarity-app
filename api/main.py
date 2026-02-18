from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.predict import router as predict_router

app = FastAPI()

# --- CORS CONFIGURATION ---
# List the exact origins (frontend URLs) that are allowed to talk to this API
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allows requests from your Next.js app
    allow_credentials=True,      # Allows cookies/auth headers if you add them later
    allow_methods=["*"],         # Allows all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],         # Allows all headers (like Content-Type)
)
# --------------------------

app.include_router(predict_router)