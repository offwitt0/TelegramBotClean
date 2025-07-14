import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from urllib.parse import quote
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from email.message import EmailMessage
import imaplib
import smtplib
import email
import openai

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ==================== LOAD ENV & CONFIG ====================
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

openai.api_key = OPENAI_API_KEY
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== LOAD DATA ====================
with open("listings.json", "r", encoding="utf-8") as f:
    listings_data = json.load(f)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "guest_kb_vectorstore", embeddings, allow_dangerous_deserialization=True
)

# ==================== UTILITIES ====================
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

# ==================== SHARED MESSAGE PROCESSING ====================
chat_history = {}

def process_message(user_message, user_id):
    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append({"role": "user", "content": user_message})

    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    links = {
        "Zamalek": generate_airbnb_link("Zamalek", checkin, checkout),
        "Maadi": generate_airbnb_link("Maadi", checkin, checkout),
        "Garden City": generate_airbnb_link("Garden City", checkin, checkout),
    }
    custom_links = "\n".join([f"[Explore {k}]({v})" for k, v in links.items()])

    listings = find_matching_listings("Cairo", 5)
    suggestions = "\n\nHere are some great options for you:\n" + "\n".join(listings) if listings else ""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{get_prompt()}\n\n{kb_context}\n{custom_links}\n{suggestions}"},
            *chat_history[user_id]
        ],
        temperature=0.7,
        max_tokens=1000
    )
    reply = response.choices[0].message.content.strip()
    chat_history[user_id].append({"role": "assistant", "content": reply})
    return reply

# ==================== TELEGRAM ====================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏨 Welcome to your vacation rental assistant! I'm here to help you find the perfect stay in Cairo, Egypt. Where would you like to travel and when?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_message = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    reply = process_message(user_message, user_id)
    await update.message.reply_text(reply)

def run_telegram():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Telegram bot running...")
    app.run_polling()

# ==================== EMAIL ====================
def check_email():
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

        print(f"📩 Email from {from_email}: {subject}")
        try:
            reply = process_message(body, from_email)
            msg_reply = EmailMessage()
            msg_reply["From"] = EMAIL_ADDRESS
            msg_reply["To"] = from_email
            msg_reply["Subject"] = f"Re: {subject}"
            msg_reply.set_content(reply)

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg_reply)
            print("✅ Email replied.")
        except Exception as e:
            print("❌ Error:", e)
    mail.logout()

def run_email():
    print("📧 Email listener running...")
    while True:
        check_email()
        time.sleep(30)

# ==================== MAIN ====================
if __name__ == "__main__":
    threading.Thread(target=run_email, daemon=True).start()
    run_telegram()
