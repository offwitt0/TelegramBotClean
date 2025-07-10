import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from urllib.parse import quote

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# ==================== BOT CODE ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏨 Welcome to your vacation rental assistant! I'm here to help you find the perfect stay in Cairo, Egypt. Where would you like to travel and when?"
    )

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "guest_kb_vectorstore", 
    embeddings,
    allow_dangerous_deserialization=True
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = str(update.effective_user.id)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Memory
    if "chat_history" not in context.chat_data:
        context.chat_data["chat_history"] = {}
    if user_id not in context.chat_data["chat_history"]:
        context.chat_data["chat_history"][user_id] = []

    context.chat_data["chat_history"][user_id].append({"role": "user", "content": user_message})

    # Load vectorstore context
    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Airbnb Link Suggestions
    today = datetime.today().date()
    checkin = today + timedelta(days=3)
    checkout = today + timedelta(days=6)

    links = {
        "Zamalek": generate_airbnb_link("Zamalek", checkin, checkout),
        "Maadi": generate_airbnb_link("Maadi", checkin, checkout),
        "Garden City": generate_airbnb_link("Garden City", checkin, checkout),
    }

    custom_links = "\n".join([f"[Explore {k}]({v})" for k, v in links.items()])

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""
{get_prompt()}

Use this internal SOP knowledge base if helpful:
{kb_context}

Some suggested areas:
{custom_links}
"""
                },
                *context.chat_data["chat_history"][user_id]
            ],
            max_tokens=1000,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        context.chat_data["chat_history"][user_id].append({"role": "assistant", "content": reply})

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Sorry, something went wrong. Please try again.")

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot starting...")
    app.run_polling()

if __name__ == '__main__':
    main()