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


# Payment
import requests

# API endpoint
url = "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation"
# Data payload
data = {
    "userName": "tonaja Mohamed",
    "email": "tonaja.mohamed@gmail.com",
    "roomType": "test",
    "checkIn": "2025-07-17T12:39:40.090Z",
    "checkOut": "2025-07-17T12:39:40.091Z",
    "numberOfGuests": 3,
    "amountInCents": 7000,
    "successfulURL": "http://localhost:3000/thanks",
    "cancelURL": "http://localhost:3000/cancel"
}
# Send the POST request
response = requests.post(url, json=data)

def Payment():
    # Check the response
    if response.status_code == 200:
        result = response.json()
        PaymentUrl = result['sessionURL']
        return PaymentUrl
    else:
        failed = 'failed to send'
        return failed

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

def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
    area_encoded = quote(area)
    return (
        f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes"
        f"?checkin={checkin}&checkout={checkout}&adults={adults}"
        f"&children={children}&infants={infants}&pets={pets}"
    )

def get_prompt():
    payment_url = Payment()
    return f"""
You are a professional, friendly, and detail-oriented guest experience assistant working for a short-term rental company in Cairo, Egypt.
Always help with questions related to vacation stays, Airbnb-style bookings, and guest policies.
Only ignore a question if it's completely unrelated to travel.
Use the internal knowledge base provided to answer questions clearly and accurately.

If the user/client wants to book the room or finalize the payment, give them this URL: {payment_url}
"""

def find_matching_listings(query, guests=2):
    query_lower = query.lower()
    query_words = query_lower.split()
    matched = []
    unmatched = []

    for listing in listings_data:
        name = listing.get("name", "").lower()
        city = listing.get("city_hint", "").lower()
        guest_ok = (listing.get("guests") or 0) >= guests

        city_match = any(word in city for word in query_words)
        name_match = any(word in name for word in query_words)

        url = listing.get("url") or f"https://anqakhans.holidayfuture.com/listings/{listing['id']}"
        rating = listing.get("rating", "No rating")
        listing_text = f"{listing['name']} (⭐ {rating})\n{url}"

        if guest_ok:
            if city_match or name_match:
                matched.append(listing_text)
            else:
                unmatched.append(listing_text)

    # Prioritize matched listings, but fallback to all available if nothing matches
    return matched if matched else unmatched


def generate_response(user_message, sender_id=None, history=None):
    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    # Search knowledge base
    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("⚙️ generating response for:", user_message)

    # Match listings using user's message (free text)
    listings = find_matching_listings(user_message, guests=2)
    booking_intent_keywords = ["book", "booking", "reserve", "reservation", "interested", "want to stay"]
    booking_intent_detected = any(kw in user_message.lower() for kw in booking_intent_keywords)

    if listings:
        # Try to get the first actual matched listing data
        matched_listing = None
        for l in listings_data:
            if l["name"] in listings[0]:
                matched_listing = l
                break

        if booking_intent_detected and matched_listing:
            listing_text = f"Great! Here's the listing you’re interested in:\n\n" \
                        f"**{matched_listing['name']} (⭐ {matched_listing.get('rating', 'N/A')})**\n{matched_listing['url']}"

            # Standard house rules
            rules_text = "\n".join([
                "• Check-in: 3:00 PM",
                "• Check-out: 12:00 PM",
                "• Pets: Not allowed",
                "• Parties: Not allowed",
                "• Smoking: Not allowed"
            ])

            suggestions = listing_text + f"\n\n📋 **House Rules:**\n{rules_text}"
        else:
            suggestions = "\n\nHere are some great options for you:\n" + "\n".join(listings)
    else:
        suggestions = "\n\nI'm sorry, I couldn't find matching listings. Please try a different area, name, or number of guests."


    # Optional: Add area links (static Airbnb-style)
    links = {
        "Zamalek": generate_airbnb_link("Zamalek", checkin, checkout),
        "Maadi": generate_airbnb_link("Maadi", checkin, checkout),
        "Garden City": generate_airbnb_link("Garden City", checkin, checkout),
    }
    custom_links = "\n".join([f"[Explore {k}]({v})" for k, v in links.items()])

    # Compose message
    chat_history = ""
    if history:
        for turn in history[-6:]:  # use only the last few turns
            role = turn["role"]
            content = turn["content"]
            chat_history += f"{role.upper()}: {content}\n"

    system_message = f"""{get_prompt()}

    Previous conversation:
    {chat_history}

    Knowledge base:
    {kb_context}

    {custom_links}
    {suggestions}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
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

                # 🔁 Load chat history
                history = load_email_history(from_email)
                history.append({"role": "user", "content": body})

                # 🧠 Generate reply with memory
                reply = generate_response(body, from_email, history)

                # ✅ Save new response to history
                history.append({"role": "assistant", "content": reply})
                save_email_history(from_email, history)

                # Send email back
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    context.chat_data["chat_history"] = {}
    context.chat_data["user_email"] = {}

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
    if user_id not in context.chat_data["user_email"]:
        if is_valid_email(user_message):
            context.chat_data["user_email"][user_id] = user_message
            save_user_email_mapping(user_id, user_message)
            await update.message.reply_text("✅ Email saved. When are you planning to travel to Cairo?")
        else:
            await update.message.reply_text("📧 Please provide a valid email address to continue.")
        return

    context.chat_data["chat_history"][user_id].append({"role": "user", "content": user_message})
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = generate_response(user_message, user_id, context.chat_data["chat_history"][user_id])
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
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000)