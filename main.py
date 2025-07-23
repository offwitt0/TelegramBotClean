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
import dateutil.parser
import calendar
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import requests
import string

# Payment
def Payment(user_name, email, room_type, checkin, checkout, number_of_guests, amountInCents):
    url = "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation"
    data = {
        "userName": user_name,
        "email": email,
        "roomType": room_type,
        "checkIn": checkin.isoformat(),
        "checkOut": checkout.isoformat(),
        "numberOfGuests": number_of_guests,
        "amountInCents": int(amountInCents),
        "successfulURL": "http://localhost:3000/thanks",
        "cancelURL": "http://localhost:3000/cancel"
    }

    print("🔍 Payload to API:", data)
    try:
        response = requests.post(url, json=data)
        print("📨 Status Code:", response.status_code)
        print("📨 Response Text:", response.text)

        if response.status_code == 200:
            session_url = response.json().get("sessionURL")
            print("✅ Stripe Session URL:", session_url)
            return session_url
        else:
            return None
    except Exception as e:
        logging.error("Payment error: %s", e)
        print("❌ Exception occurred:", e)
        return None

def extract_dates_from_message(message):
    try:
        # Example: "from 20 to 25 Aug" or "20 to 25 Aug"
        pattern = r'(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(\w{3,9})'
        match = re.search(pattern, message.lower())
        if match:
            day1 = int(match.group(1))
            day2 = int(match.group(2))
            month_str = match.group(3)

            # Try to convert month to number
            try:
                month = list(calendar.month_name).index(month_str.capitalize())
                if month == 0:
                    month = list(calendar.month_abbr).index(month_str.capitalize())
            except ValueError:
                return None, None

            current_year = datetime.now().year
            checkin = datetime(current_year, month, day1)
            checkout = datetime(current_year, month, day2)
            if checkin < checkout:
                return checkin.date(), checkout.date()
    except:
        pass
    return None, None

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

# def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
#     area_encoded = quote(area)
#     return (
#         f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes"
#         f"?checkin={checkin}&checkout={checkout}&adults={adults}"
#         f"&children={children}&infants={infants}&pets={pets}"
#     )

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

        # Strong match: query contains exact city name
        if any(q == city_lower for q in query_words):
            matched.append(listing_text)
        # Medium match: query contains words from name or city
        elif any(q in name_lower or q in city_lower for q in query_words):
            fallback.append(listing_text)

    if matched:
        return matched[:5]
    elif fallback:
        return fallback[:3]
    else:
        return []

def generate_response(user_message, sender_id=None, history=None):
    checkin, checkout = extract_dates_from_message(user_message)
    if not checkin or not checkout:
        today = datetime.today().date()
        checkin = today + timedelta(days=3)
        checkout = today + timedelta(days=6)
        Days = (checkout - checkin).days
    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    listings = find_matching_listings(user_message, guests=2)
    booking_intent_keywords = ["book", "booking", "reserve", "reservation", "interested", "want to stay"]
    booking_intent_detected = any(kw in user_message.lower() for kw in booking_intent_keywords)

    matched_listing = next(
        (l for l in listings_data if l["name"].lower() in user_message.lower()),
        None
    )
    user_email = sender_id if sender_id and "@" in sender_id else "guest@example.com"

    payment_url = None
    suggestions = ""

    if booking_intent_detected and matched_listing:
        amount = matched_listing.get("price", 7000)
        payment_url = Payment(
            user_name="Guest",
            email=user_email,
            room_type=matched_listing["name"],
            checkin=checkin,
            checkout=checkout,
            number_of_guests=2,
            amountInCents=int(amount * 100)
        )
        listing_text = f"Great to hear that you're ready to proceed with the booking!\nTo finalize your reservation for the {matched_listing['name']} in Cairo, Egypt, please complete the payment through this secure link:\n{payment_url}\n\n"
        rules_text = "\n".join([
            "• Check-in: 3:00 PM",
            "• Check-out: 12:00 PM",
            "• Pets: Not allowed",
            "• Parties: Not allowed",
            "• Smoking: Not allowed"
        ])
        suggestions = listing_text + f"📋 House Rules:\n{rules_text}"

    elif listings:
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
    now = datetime.utcnow()

    # Setup default storage
    context.chat_data.setdefault("chat_history", {})
    context.chat_data.setdefault("user_email", {})
    context.chat_data.setdefault("last_active", {})
    context.chat_data.setdefault("all_messages", {})

    # Inactivity check: 2 minutes
    last_active = context.chat_data["last_active"].get(user_id)
    if last_active and (now - last_active).total_seconds() > 120:
        # 🧹 Delete all stored message IDs
        for msg_id in context.chat_data["all_messages"].get(user_id, []):
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg_id)
            except Exception as e:
                logging.warning(f"⚠️ Failed to delete message {msg_id}: {e}")

        # 🧼 Clear session data
        context.chat_data["chat_history"].pop(user_id, None)
        context.chat_data["user_email"].pop(user_id, None)
        context.chat_data["last_active"].pop(user_id, None)
        context.chat_data["all_messages"].pop(user_id, None)

        msg = await update.message.reply_text("🕒 Your session has been reset due to inactivity. Please enter your email to get started.")
        context.chat_data["all_messages"].setdefault(user_id, []).append(msg.message_id)
        return

    # ⏱️ Update last activity timestamp
    context.chat_data["last_active"][user_id] = now

    # 💬 Store the user's message ID to delete later
    context.chat_data["all_messages"].setdefault(user_id, []).append(update.message.message_id)

    # ✅ Handle email collection if needed
    if user_id not in context.chat_data["user_email"]:
        if is_valid_email(user_message):
            context.chat_data["user_email"][user_id] = user_message
            save_user_email_mapping(user_id, user_message)
            reply = await update.message.reply_text("✅ Email saved. When are you planning to travel to Cairo?")
            context.chat_data["all_messages"][user_id].append(reply.message_id)
        else:
            reply = await update.message.reply_text("📧 Please provide a valid email address to continue.")
            context.chat_data["all_messages"][user_id].append(reply.message_id)
        return

    # ✍️ Add message to chat history
    context.chat_data["chat_history"].setdefault(user_id, []).append({"role": "user", "content": user_message})

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply_text = generate_response(
            user_message,
            context.chat_data["user_email"][user_id],
            context.chat_data["chat_history"][user_id]
        )
        reply_msg = await update.message.reply_text(reply_text)
        context.chat_data["all_messages"][user_id].append(reply_msg.message_id)
        context.chat_data["chat_history"][user_id].append({"role": "assistant", "content": reply_text})
    except Exception as e:
        err = await update.message.reply_text("❌ Bot error")
        context.chat_data["all_messages"][user_id].append(err.message_id)
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