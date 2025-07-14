import os
import json
import logging
import imaplib
import smtplib
import email
from datetime import datetime, timedelta
from urllib.parse import quote
from email.message import EmailMessage
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# =================== LOAD CONFIG ===================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is not set in environment variables.")

# =================== INIT ===================
openai = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True)

with open("listings.json", "r", encoding="utf-8") as f:
    listings_data = json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================== CORE FUNCTIONS ===================
def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
    area_encoded = quote(area)
    return f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes?checkin={checkin}&checkout={checkout}&adults={adults}&children={children}&infants={infants}&pets={pets}"

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

# =================== TELEGRAM ===================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\ud83c\udfe8 Welcome! Tell me your vacation needs in Cairo.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    reply = generate_response(user_message)
    await update.message.reply_text(reply)

telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

# =================== EMAIL ===================
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

            print(f"\ud83d\udce9 Email from {from_email}: {subject}")
            reply = generate_response(body)
            send_email(from_email, subject, reply)
            print("\u2705 Email replied.")

        mail.logout()
    except Exception as e:
        logger.error(f"Email Error: {e}")

@app.on_event("startup")
async def startup_event():
    import asyncio
    from threading import Thread

    def run_email_loop():
        import time
        print("\ud83d\udce7 Email listener running...")
        while True:
            check_email()
            time.sleep(15)  # Check every 15 seconds

    def run_telegram_loop():
        print("\ud83e\uddf0 Telegram bot running...")
        telegram_app.run_polling()

    Thread(target=run_email_loop).start()
    Thread(target=run_telegram_loop).start()