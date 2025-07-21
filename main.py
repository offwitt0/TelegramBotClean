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
import sys
sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
<<<<<<< HEAD
import requests
import string

# Payment function updated to accept dynamic user data
def Payment(user_data):
    # API endpoint
    url = "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation"
    data = {
        "userName": user_data.get("userName", "Guest User"),
        "email": user_data.get("email", "default@example.com"),
        "roomType": user_data.get("roomType", "standard"),
        "checkIn": user_data.get("checkIn", (datetime.today() + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
        "checkOut": user_data.get("checkOut", (datetime.today() + timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
        "numberOfGuests": user_data.get("numberOfGuests", 2),
        "amountInCents": user_data.get("amountInCents", 7000),
        "successfulURL": "http://localhost:3000/thanks",
        "cancelURL": "http://localhost:3000/cancel"
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("sessionURL")
        else:
            return None
    except Exception as e:
        logging.error("Payment error: %s", e)
        return None
=======
import logging 
>>>>>>> 5decdd7 (try again103)

# ================== ENV & CONFIG ==================
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

client = OpenAI(api_key=OPENAI_API_KEY)

def load_email_history(email_address):
    try:
        with open("email_history.json", "r") as f:
            history = json.load(f)
        return history.get(email_address, [])
    except FileNotFoundError:
        return []

def save_email_history(email_address, history):
    try:
        with open("email_history.json", "r") as f:
            all_history = json.load(f)
    except FileNotFoundError:
        all_history = {}
    all_history[email_address] = history
    with open("email_history.json", "w") as f:
        json.dump(all_history, f, indent=2)

# ================== AIRBNB & DATA ==================
with open("listings.json", "r", encoding="utf-8") as f:
    listings_data = json.load(f)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True)

def get_prompt(payment_url=None):
    base = """
    You are a professional, friendly, and detail-oriented guest experience assistant working for a short-term rental company in Cairo, Egypt.
    Always help with questions related to vacation stays, Airbnb-style bookings, and guest policies.
    Only ignore a question if it's completely unrelated to travel.
    Use the internal knowledge base provided to answer questions clearly and accurately.
    """
    if payment_url:
        base += f"\n\nIf the user/client wants to book the room or finalize the payment, give them this exact URL without modifying it:\n{payment_url}"
    return base

def find_matching_listings(query, guests=2):
    query_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
    query_words = query_clean.split()

    matched = []
    fallback = []

    for listing in listings_data:
        name = listing.get("name", "")
        city = listing.get("city_hint", "")
        guest_ok = (listing.get("guests") or 0) >= guests

        if not guest_ok:
            continue

        name_lower = name.lower()
        city_lower = city.lower()
        url = listing.get("url") or f"https://anqakhans.holidayfuture.com/listings/{listing['id']}"
        rating = listing.get("rating", "No rating")
        listing_text = f"{name} (⭐ {rating})\n{url}"

        if any(q == city_lower for q in query_words):
            matched.append(listing_text)
        elif any(q in name_lower or q in city_lower for q in query_words):
            fallback.append(listing_text)

    if matched:
        return matched[:5]
    elif fallback:
        return fallback[:3]
    else:
        return []

def generate_response(user_message, sender_id=None, history=None, user_data=None):
    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    listings = find_matching_listings(user_message, guests=user_data.get("numberOfGuests", 2) if user_data else 2)
    booking_intent_keywords = ["book", "booking", "reserve", "reservation", "interested", "want to stay"]
    booking_intent_detected = any(kw in user_message.lower() for kw in booking_intent_keywords)

    payment_url = Payment(user_data) if booking_intent_detected and user_data else None

    suggestions = ""
    if listings:
        matched_listing = next((l for l in listings_data if l["name"] in listings[0]), None)

        if booking_intent_detected and matched_listing:
            listing_text = f"Great to hear that you're ready to proceed with the booking!\nTo finalize your reservation for the {matched_listing['name']} in Cairo, Egypt, please complete the payment through this secure link:\n{payment_url}\n\n"
            rules_text = "\n".join([
                "• Check-in: 3:00 PM",
                "• Check-out: 12:00 PM",
                "• Pets: Not allowed",
                "• Parties: Not allowed",
                "• Smoking: Not allowed"
            ])
            suggestions = listing_text + f"📋 House Rules:\n{rules_text}"
        else:
            suggestions = "\n\nHere are some great options for you:\n" + "\n".join(listings)
    else:
        suggestions = "\n\nI'm sorry, I couldn't find matching listings. Please try a different area, name, or number of guests."

    chat_history = ""
    if history:
        for turn in history[-6:]:
            chat_history += f"{turn['role'].upper()}: {turn['content']}\n"

    system_message = f"""{get_prompt(payment_url)}
    Previous conversation:
    {chat_history}

    Knowledge base:
    {kb_context}

    {suggestions}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"rrh": "user", "content": user_message}
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
                    body = msg.get_payload(decode=True).decode麻

                print(f"📩 Email from {from_email}: {subject}")

                history = load_email_history(from_email)
                history.append({"role": "user", "content": body})

                # Extract user data from email body (simplified example)
                user_data = {
                    "email": from_email,
                    "userName": from_email.split("@")[0],
                    "roomType": "standard",
                    "numberOfGuests": 2
                }
                reply = generate_response(body, from_email, history, user_data)

                history.append({"role": "assistant", "content": reply})
                save_email_history(from_email, history)

                send_email(from_email, subject, reply)
                print("✅ Email replied.")
            mail.logout()
        except Exception as e:
            print("❌ Email error:", e)
        await asyncio.sleep(5)

# ================== TELEGRAM ==================
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

def is_valid_email(email_str: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email_str) is not None

def save_user_email_mapping(user_id: str, email_address: str):
    mapping_path = "user_mapping.json"
    try:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    except FileNotFoundError:
        mapping = {}
    mapping[user_id] = email_address
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
<<<<<<< HEAD

=======
 
 
def save_payment_url(user_id: str, payment_url: str):
    path = "payment_urls.json"
    try:
        with open(path, "r") as f:
            urls = json.load(f)
    except FileNotFoundError:
        urls = {}
    urls[user_id] = payment_url
    with open(path, "w") as f:
        json.dump(urls, f, indent=2)
 
 
async def send_email_to_api(user_id: str, email: str):
    url = "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation"
    payload = {
        "userName": "tonaja Mohamed",
        "email": email,
        "roomType": "test",
        "checkIn": "2025-07-17T12:39:40.090Z",
        "checkOut": "2025-07-17T12:39:40.091Z",
        "numberOfGuests": 3,
        "amountInCents": 7000,
        "successfulURL": "http://localhost:3000/thanks",
        "cancelURL": "http://localhost:3000/cancel"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation",
            json=payload
        )
        result = response.json()  # ✅ Do NOT use `await`
        return result

 
 
>>>>>>> 5decdd7 (try again103)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    context.chat_data["chat_history"] = {}
    context.chat_data["user_email"] = {}
    context.chat_data["user_data"] = {}

    await update.message.reply_text("🏨 Welcome! Please enter your email to get started.")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)

    if "chat_history" not in context.chat_data:
        context.chat_data["chat_history"] = {}
    if user_id not in context.chat_data["chat_history"]:
        context.chat_data["chat_history"][user_id] = []

    if "user_email" not in context.chat_data:
        context.chat_data["user_email"] = {}
    if "user_data" not in context.chat_data:
        context.chat_data["user_data"] = {}

    if user_id not in context.chat_data["user_email"]:
        if is_valid_email(user_message):
            context.chat_data["user_email"][user_id] = user_message
            save_user_email_mapping(user_id, user_message)
            context.chat_data["user_data"][user_id] = {
                "email": user_message,
                "userName": user_message.split("@")[0],
                "roomType": "standard",
                "numberOfGuests": 2
            }
            await update.message.reply_text("✅ Email saved. When are you planning to travel to Cairo? (e.g., 'I want to book a room for 3 guests from 2025-07-20 to 2025-07-25')")
        else:
            await update.message.reply_text("📧 Please provide a valid email address to continue.")
        return

    # Parse user message for booking details
    user_data = context.chat_data["user_data"][user_id]
    if "book" in user_message.lower() or "reserve" in user_message.lower():
        # Extract dates and number of guests from message (simplified regex for example)
        date_pattern = r"(\d{4}-\d{2}-\d{2})"
        guest_pattern = r"(\d+)\s*(guest|guests)"
        dates = re.findall(date_pattern, user_message)
        guests = re.search(guest_pattern, user_message)

        if dates and len(dates) >= 2:
            user_data["checkIn"] = dates[0] + "T12:00:00.000Z"
            user_data["checkOut"] = dates[1] + "T12:00:00.000Z"
        if guests:
            user_data["numberOfGuests"] = int(guests.group(1))

    context.chat_data["chat_history"][user_id].append({"role": "user", "content": user_message})
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = generate_response(user_message, user_id, context.chat_data["chat_history"][user_id], user_data)
        await update.message.reply_text(reply)
        context.chat_data["chat_history"][user_id].append({"role": "assistant", "content": reply})
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
<<<<<<< HEAD
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000)
=======
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000)
    asyncio.run(send_email_to_api("123", "test@example.com"))

 
>>>>>>> 5decdd7 (try again103)
