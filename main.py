import os
import json
import logging
import asyncio
import imaplib
import smtplib
import email
from email.message import EmailMessage
from datetime import datetime, timedelta
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ================== ENV & CONFIG ==================
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== AIRBNB & DATA ==================
with open("listings.json", "r", encoding="utf-8") as f:
    listings_data = json.load(f)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True)

def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
    area_encoded = quote(area)
    return (
        f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes"
        f"?checkin={checkin}&checkout={checkout}&adults={adults}"
        f"&children={children}&infants={infants}&pets={pets}"
    )

def get_prompt():
    return """
You are a professional, friendly, and detail-oriented guest experience assistant working for a short-term rental company in Cairo, Egypt.
Always help with questions related to vacation stays, Airbnb-style bookings, and guest policies.
Only ignore a question if it's completely unrelated to travel.
Use the internal knowledge base provided to answer questions clearly and accurately.
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
    print("⚙️ generating response for:", user_message)

    links = {
        "Zamalek": generate_airbnb_link("Zamalek", checkin, checkout),
        "Maadi": generate_airbnb_link("Maadi", checkin, checkout),
        "Garden City": generate_airbnb_link("Garden City", checkin, checkout),
    }
    custom_links = "\n".join([f"[Explore {k}]({v})" for k, v in links.items()])
    suggestions = "\n\nHere are some great options:\n" + "\n".join(find_matching_listings("Cairo", 4))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{get_prompt()}\n\n{kb_context}\n\n{custom_links}\n{suggestions}"},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ================== EMAIL ==================
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

async def check_email_loop():
    while True:
        try:
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select("inbox")
            _, messages = mail.search(None, '(UNSEEN)')
            for num in messages[0].split():
                _, msg_data = mail.fetch(num, '(RFC822)')
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

                print(f"📩 Email from {from_email}: {subject}")
                reply = generate_response(body)
                send_email(from_email, subject, reply)
                print("✅ Email replied.")
            mail.logout()
        except Exception as e:
            print("❌ Email error:", e)
        await asyncio.sleep(30)

# ================== TELEGRAM ==================
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("✅ Telegram message received:", update.message.text)
    await update.message.reply_text("🏨 Welcome! When are you planning to travel to Cairo?")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)

    if "guest_sessions" not in context.chat_data:
        context.chat_data["guest_sessions"] = {}

    if user_id not in context.chat_data["guest_sessions"]:
        context.chat_data["guest_sessions"][user_id] = {
            "state": "start",
            "unit": None,
            "checkin": None,
            "checkout": None,
            "guests": None,
            "email": None
        }

    session = context.chat_data["guest_sessions"][user_id]
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        if session["state"] == "start":
            session["state"] = "awaiting_unit"
            await update.message.reply_text("🏘️ Thanks for reaching out! Which unit are you interested in?")
            return

        elif session["state"] == "awaiting_unit":
            session["unit"] = user_message
            session["state"] = "awaiting_checkin"
            await update.message.reply_text("📅 Great! What is your check-in date? (e.g., 2025-08-12)")
            return

        elif session["state"] == "awaiting_checkin":
            session["checkin"] = user_message
            session["state"] = "awaiting_checkout"
            await update.message.reply_text("📅 And your check-out date? (e.g., 2025-08-15)")
            return

        elif session["state"] == "awaiting_checkout":
            session["checkout"] = user_message
            session["state"] = "awaiting_guests"
            await update.message.reply_text("👥 How many guests will be staying?")
            return

        elif session["state"] == "awaiting_guests":
            session["guests"] = user_message
            session["state"] = "inquiry_complete"

            summary = (
                f"✅ Here's what I got:\n"
                f"• Unit: {session['unit']}\n"
                f"• Check-in: {session['checkin']}\n"
                f"• Check-out: {session['checkout']}\n"
                f"• Guests: {session['guests']}\n\n"
                f"I'll now check availability and get back to you!"
            )
            await update.message.reply_text(summary)
            return

        else:
            reply = generate_response(user_message)
            await update.message.reply_text(reply)

    except Exception as e:
        await update.message.reply_text("❌ Bot error")
        logging.error(e)

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ================== FASTAPI ==================
fastapi_app = FastAPI()

@fastapi_app.on_event("startup")
async def start_all():
    print("📧 Email listener running...")
    asyncio.create_task(check_email_loop())

    print("🤖 Telegram bot initializing...")
    await app.initialize()
    await app.start()
    asyncio.create_task(app.updater.start_polling())

@fastapi_app.on_event("shutdown")
async def shutdown_all():
    print("⛔ Shutting down bot...")
    await app.stop()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000)