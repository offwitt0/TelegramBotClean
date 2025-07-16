# Full updated `main.py` integrating:
# - Smart customer service behavior
# - Greeting, Q&A, and booking detection
# - Telegram + Email support

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

import pandas as pd
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

# ================== Load Excel & Bookings ==================
EXCEL_PATH = "AnQa.xlsx"
BOOKINGS_PATH = "bookings.json"
listings_df = pd.read_excel(EXCEL_PATH)
if not os.path.exists(BOOKINGS_PATH):
    with open(BOOKINGS_PATH, "w") as f:
        json.dump({}, f)

confirmed_sessions = {}  # session_id -> {unit_name, dates}

# ================== Booking Logic ==================
def is_available(unit_name, start_date, end_date):
    with open(BOOKINGS_PATH, "r") as f:
        bookings = json.load(f)
    booked_ranges = bookings.get(unit_name, [])
    for r in booked_ranges:
        booked_start = datetime.strptime(r[0], "%Y-%m-%d").date()
        booked_end = datetime.strptime(r[1], "%Y-%m-%d").date()
        if not (end_date <= booked_start or start_date >= booked_end):
            return False
    return True

def save_booking(unit_name, start_date, end_date):
    with open(BOOKINGS_PATH, "r") as f:
        bookings = json.load(f)
    bookings.setdefault(unit_name, []).append([str(start_date), str(end_date)])
    with open(BOOKINGS_PATH, "w") as f:
        json.dump(bookings, f, indent=2)

# ================== Property Filtering ==================
def filter_properties(requirements: dict):
    df = listings_df.copy()
    if requirements.get("area"):
        df = df[df["Area"].str.lower() == requirements["area"].lower()]
    if "guests" in requirements:
        df = df[df["Guests"] >= requirements["guests"]]
    if "bathrooms" in requirements:
        df = df[df["Bathrooms #"] >= requirements["bathrooms"]]
    if requirements.get("pets"):
        df = df[df["Luggage"] == "Yes"]
    if "start_date" in requirements and "end_date" in requirements:
        df = df[df["Unit Name"].apply(lambda u: is_available(u, requirements["start_date"], requirements["end_date"]))]
    return df.head(3)

def generate_airbnb_link(area, checkin, checkout, adults=2):
    area_encoded = quote(area)
    return f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes?checkin={checkin}&checkout={checkout}&adults={adults}"

# ================== NLP & Booking Engine ==================
def parse_requirements(message):
    message = message.lower()
    req = {"guests": 0, "bathrooms": 0, "area": None, "pets": False}
    if "zamalek" in message:
        req["area"] = "Zamalek"
    if "maadi" in message:
        req["area"] = "Maadi"
    if "garden city" in message:
        req["area"] = "Garden City"
    if "pets" in message:
        req["pets"] = True
    for g in range(1, 11):
        if f"{g} adult" in message or f"{g} guests" in message:
            req["guests"] = g
    for b in range(1, 6):
        if f"{b} bathroom" in message:
            req["bathrooms"] = b
    req["start_date"] = datetime.today().date() + timedelta(days=5)
    req["end_date"] = req["start_date"] + timedelta(days=4)
    return req

def detect_intent(message):
    message = message.lower()
    greetings = ["hi", "hello", "hey"]
    if any(g in message for g in greetings):
        return "greeting"
    if any(x in message for x in ["book", "stay", "rent", "apartment", "hotel"]):
        return "booking"
    return "question"

def generate_response(user_message, session_id):
    intent = detect_intent(user_message)

    if intent == "greeting":
        return "👋 Hello! How can I assist you today regarding your stay in Cairo?"

    if intent == "question":
        relevant_docs = vectorstore.similarity_search(user_message, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful hotel assistant in Cairo."},
                {"role": "user", "content": f"{context}\n\nUser: {user_message}"}
            ]
        )
        return completion.choices[0].message.content.strip()

    req = parse_requirements(user_message)
    matching = filter_properties(req)
    confirmed_sessions[session_id] = {"candidates": list(matching["Unit Name"]) if not matching.empty else [], **req}
    if matching.empty:
        return "❌ Sorry, no available listings match your request."
    msgs = []
    for _, row in matching.iterrows():
        unit = row["Unit Name"]
        link = generate_airbnb_link(row["Area"], req["start_date"], req["end_date"], req["guests"])
        msgs.append(f"🏡 *{unit}* in {row['Area']}\n- Guests: {int(row['Guests'])}, Bathrooms: {row['Bathrooms #']}\n[View Listing]({link})")
    return "\n\n".join(msgs) + "\n\nPlease reply with the *exact* name of the apartment to confirm booking."

def try_confirm_booking(session_id, user_message):
    data = confirmed_sessions.get(session_id)
    if not data:
        return None
    for name in data.get("candidates", []):
        if name.lower() in user_message.lower():
            save_booking(name, data["start_date"], data["end_date"])
            return f"✅ Booking confirmed for *{name}* from {data['start_date']} to {data['end_date']}"
    return None

# ================== Email ==================
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
                confirmation = try_confirm_booking(from_email, body)
                reply = confirmation or generate_response(body, from_email)
                send_email(from_email, subject, reply)
                print("✅ Email replied.")
            mail.logout()
        except Exception as e:
            print("❌ Email error:", e)
        await asyncio.sleep(30)

# ================== Telegram ==================
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏨 Welcome! How can I help you with your stay in Cairo?")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)
    print("📲 Telegram message received:", user_message)
    confirmation = try_confirm_booking(user_id, user_message)
    reply = confirmation or generate_response(user_message, user_id)
    await update.message.reply_text(reply, parse_mode="Markdown")

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ================== FastAPI ==================
fastapi_app = FastAPI()

@fastapi_app.on_event("startup")
async def startup():
    print("📧 Email listener running...")
    asyncio.create_task(check_email_loop())
    print("🤖 Telegram bot initializing...")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000)