import os
import json
import logging
import asyncio
import imaplib
import smtplib
import email
import re
from email.message import EmailMessage
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load environment variables
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ========== Vectorstore loading ==========
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True)

# ========== Telegram Setup ==========
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏨 Welcome to AnQa! How can I assist you today with your travel or booking needs in Cairo?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)

    # Search the knowledge base
    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a smart assistant for AnQa Booking. You help users with:
- General questions
- Booking-related inquiries
- Policies, services, areas in Cairo

Use this context:
{kb_context}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )

    reply = response.choices[0].message.content.strip()
    await update.message.reply_text(reply, parse_mode="Markdown")

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ========== FastAPI App ==========
fastapi_app = FastAPI()

@fastapi_app.on_event("startup")
async def startup():
    await app.initialize()
    await app.start()
    asyncio.create_task(app.updater.start_polling())

@fastapi_app.on_event("shutdown")
async def shutdown():
    await app.stop()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000, reload=True)
