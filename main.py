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
import requests

# ================== CONFIGURATION ==================
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ================== DATA STORES ==================
def load_data():
    """Load all required data stores"""
    with open("listings.json", "r", encoding="utf-8") as f:
        listings = json.load(f)
    
    try:
        vectorstore = FAISS.load_local(
            "guest_kb_vectorstore", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except:
        vectorstore = None
        logging.warning("Vectorstore not loaded")
    
    return listings, vectorstore

listings_data, vectorstore = load_data()

# ================== PAYMENT HANDLER ==================
def generate_payment_link(user_data: dict):
    """Generate payment link with user-specific data"""
    url = "https://subscriptionsmanagement.fastautomate.com/api/Payments/reservation"
    
    payload = {
        "userName": user_data.get("name", "Guest"),
        "email": user_data.get("email"),
        "roomType": user_data.get("room_type", "Standard Room"),
        "checkIn": user_data.get("check_in", datetime.now().isoformat()),
        "checkOut": user_data.get("check_out", (datetime.now() + timedelta(days=3)).isoformat()),
        "numberOfGuests": user_data.get("guests", 2),
        "amountInCents": int(user_data.get("amount", 70) * 100),
        "successfulURL": f"{os.getenv('BASE_URL')}/thanks",
        "cancelURL": f"{os.getenv('BASE_URL')}/cancel"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get('sessionURL')
    except requests.exceptions.RequestException as e:
        logging.error(f"Payment API error: {str(e)}")
        return None

# ================== EMAIL HISTORY ==================
def load_email_history(email_address):
    try:
        with open("email_history.json", "r") as f:
            history = json.load(f)
        return history.get(email_address, [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_email_history(email_address, history):
    try:
        with open("email_history.json", "r") as f:
            all_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_history = {}
    
    all_history[email_address] = history[-20:]  # Keep only last 20 messages
    with open("email_history.json", "w") as f:
        json.dump(all_history, f, indent=2)

# ================== LISTING MANAGEMENT ==================
def generate_airbnb_link(area, checkin, checkout, adults=2, children=0, infants=0, pets=0):
    area_encoded = quote(area)
    return (
        f"https://www.airbnb.com/s/Cairo--{area_encoded}/homes"
        f"?checkin={checkin}&checkout={checkout}&adults={adults}"
        f"&children={children}&infants={infants}&pets={pets}"
    )

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

        url = listing.get("url") or f"https://example.com/listings/{listing['id']}"
        rating = listing.get("rating", "No rating")
        listing_text = f"{listing['name']} (⭐ {rating})\n{url}"

        if guest_ok:
            if city_match or name_match:
                matched.append(listing_text)
            else:
                unmatched.append(listing_text)

    return matched if matched else unmatched[:5]  # Return max 5 unmatched if no matches

# ================== RESPONSE GENERATION ==================
def get_system_prompt():
    """Base system prompt without payment info"""
    return """
You are a professional, friendly guest experience assistant for short-term rentals in Cairo.
Always be helpful with questions about vacation stays, bookings, and policies.
Provide clear, accurate information using the knowledge base.
For booking requests, ask for: check-in/out dates, guest count, and room preference.
"""

def generate_response(user_message, sender_id=None, history=None):
    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    # Search knowledge base
    kb_context = ""
    if vectorstore:
        relevant_docs = vectorstore.similarity_search(user_message, k=3)
        kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Check for booking intent
    booking_intent = any(kw in user_message.lower() for kw in 
                        ["book", "booking", "reserve", "reservation", "pay", "payment"])

    # Find matching listings
    listings = find_matching_listings(user_message)
    listing_info = "\n\nAvailable listings:\n" + "\n".join(listings) if listings else ""

    # Generate area links
    areas = ["Zamalek", "Maadi", "Garden City"]
    area_links = "\n".join(
        f"[Explore {area}]({generate_airbnb_link(area, checkin, checkout)})" 
        for area in areas
    )

    # Build conversation history
    chat_history = ""
    if history:
        for turn in history[-6:]:  # Last 3 exchanges
            role = turn["role"]
            content = turn["content"][:500]  # Truncate long messages
            chat_history += f"{role.upper()}: {content}\n"

    # Prepare messages for OpenAI
    messages = [
        {"role": "system", "content": get_system_prompt() + f"\nKnowledge:\n{kb_context}"},
        *[{"role": msg["role"], "content": msg["content"]} for msg in (history or [])[-6:]],
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        
        # If booking intent detected and we have user email
        if booking_intent and sender_id and "@" in sender_id:
            listing = listings[0] if listings else "Standard Room"
            payment_url = generate_payment_link({
                "email": sender_id,
                "room_type": listing,
                "check_in": checkin.isoformat(),
                "check_out": checkout.isoformat(),
                "guests": 2,
                "amount": 70
            })
            
            if payment_url:
                reply += f"\n\n[Proceed to Payment]({payment_url})"
            else:
                reply += "\n\n⚠️ We couldn't generate a payment link. Please contact support."

        return reply + f"\n\n{area_links}{listing_info}"
    
    except Exception as e:
        logging.error(f"OpenAI error: {str(e)}")
        return "I'm having trouble processing your request. Please try again later."

# ================== EMAIL HANDLER ==================
def send_email(to_email, subject, body):
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg["Subject"] = f"Re: {subject}"
        msg.set_content(body)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        logging.error(f"Email send error: {str(e)}")
        return False

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

                logging.info(f"New email from {from_email}: {subject[:50]}...")

                # Process email
                history = load_email_history(from_email)
                history.append({"role": "user", "content": body})
                
                reply = generate_response(body, from_email, history)
                history.append({"role": "assistant", "content": reply})
                
                if send_email(from_email, subject, reply):
                    save_email_history(from_email, history)
                    logging.info(f"Replied to {from_email}")
                
        except Exception as e:
            logging.error(f"Email loop error: {str(e)}")
        finally:
            await asyncio.sleep(30)

# ================== TELEGRAM HANDLER ==================
def is_valid_email(email_str: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email_str) is not None

def save_user_mapping(user_id: str, email: str):
    try:
        with open("user_mapping.json", "r") as f:
            mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        mapping = {}
    
    mapping[user_id] = email
    with open("user_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏨 Welcome to Cairo Stays!\n\n"
        "Please share your email address to get started:"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = update.message.text
    
    # Initialize chat data if not exists
    if 'chat_data' not in context.chat_data:
        context.chat_data['chat_data'] = {}
    if user_id not in context.chat_data['chat_data']:
        context.chat_data['chat_data'][user_id] = {
            'history': [],
            'email': None
        }
    
    chat_data = context.chat_data['chat_data'][user_id]
    
    # Handle email collection
    if not chat_data['email'] and is_valid_email(text):
        chat_data['email'] = text
        save_user_mapping(user_id, text)
        await update.message.reply_text(
            "✅ Thank you! When are you planning to visit Cairo?"
        )
        return
    
    # Add user message to history
    chat_data['history'].append({"role": "user", "content": text})
    
    # Generate response
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    reply = generate_response(text, chat_data['email'], chat_data['history'])
    
    # Add assistant response to history
    chat_data['history'].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)

# ================== FASTAPI APP ==================
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.on_event("startup")
async def startup():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting services...")
    
    # Start email checker
    asyncio.create_task(check_email_loop())
    
    # Initialize Telegram bot
    telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    await telegram_app.initialize()
    await telegram_app.start()
    asyncio.create_task(telegram_app.updater.start_polling())

@fastapi_app.on_event("shutdown")
async def shutdown():
    logging.info("Shutting down services...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)