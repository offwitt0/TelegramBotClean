import os
import logging
import json
import imaplib
import smtplib
import email
from email.message import EmailMessage
from datetime import datetime, timedelta
from dotenv import load_dotenv
from urllib.parse import quote
from fastapi import FastAPI
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
import asyncio
from contextlib import asynccontextmanager

# Load env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

openai.api_key = OPENAI_API_KEY

# Load listings
with open("listings.json", "r", encoding="utf-8") as f:
    listings_data = json.load(f)

# Vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True)

# Shared utils
def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
    area_encoded = quote(area)
    return (
        f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes"
        f"?checkin={checkin}&checkout={checkout}"
        f"&adults={adults}&children={children}&infants={infants}&pets={pets}"
    )

def get_prompt():
    return """
You are a professional, friendly, and detail-oriented guest experience assistant working for a short-term rental company in Cairo, Egypt.

Always help with questions related to vacation stays, Airbnb-style bookings, and guest policies.

Only ignore a question if it's completely unrelated to travel (e.g., programming, politics, etc).

Use the internal knowledge base provided to answer questions clearly and accurately. Be warm and helpful.
"""

def find_matching_listings(city, guests):
    results = []
    for listing in listings_data:
        if listing["city_hint"].lower() == city.lower() and listing["guests"] >= guests:
            url = listing.get("url") or f"https://anqakhans.holidayfuture.com/listings/{listing['id']}"
            results.append(f"{listing['name']} (⭐ {listing['rating']})\n{url}")
        if len(results) >= 3:
            break
    return results

def generate_response(user_message):
    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    links = {
        "Zamalek": generate_airbnb_link("Zamalek", checkin, checkout),
        "Maadi": generate_airbnb_link("Maadi", checkin, checkout),
        "Garden City": generate_airbnb_link("Garden City", checkin, checkout),
    }
    custom_links = "\n".join([f"[Explore {k}]({v})" for k, v in links.items()])

    listings = find_matching_listings("Cairo", 5)
    suggestions = "\n\nHere are some great options for you:\n" + "\n".join(listings) if listings else ""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"{get_prompt()}\n\nUse this context if helpful:\n{kb_context}\n\n{custom_links}\n{suggestions}"
            },
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ========== EMAIL BOT ==========
def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = f"Re: {subject}"
    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def check_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN)')
        for num in messages[0].split():
            typ, msg_data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            from_email = email.utils.parseaddr(msg["From"])[1]
            subject = msg["Subject"]
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
            else:
                body = msg.get_payload(decode=True).decode()

            print(f"📩 Received from {from_email}: {subject}")
            try:
                reply = generate_response(body)
                send_email(from_email, subject, reply)
                print("✅ Replied.")
            except Exception as e:
                print("❌ Error:", e)
        mail.logout()
    except Exception as e:
        print("❌ Email loop error:", e)

async def run_email_loop():
    print("📧 Email listener running...")
    while True:
        check_email()
        await asyncio.sleep(10)

# ========== TELEGRAM BOT ==========
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏨 Welcome to your vacation rental assistant! Where would you like to travel and when?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if "chat_history" not in context.chat_data:
        context.chat_data["chat_history"] = {}
    if user_id not in context.chat_data["chat_history"]:
        context.chat_data["chat_history"][user_id] = []

    context.chat_data["chat_history"][user_id].append({"role": "user", "content": user_message})
    try:
        reply = generate_response(user_message)
        await update.message.reply_text(reply)
        context.chat_data["chat_history"][user_id].append({"role": "assistant", "content": reply})
    except Exception as e:
        logging.error(f"❌ Telegram error: {e}")
        await update.message.reply_text("❌ Sorry, something went wrong.")

async def run_telegram_bot():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Telegram bot running...")
    await app.run_polling()

# ========== FASTAPI APP ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(run_telegram_bot())
    asyncio.create_task(run_email_loop())
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok"}