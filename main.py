import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_prompt():
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    return f"""
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
    allow_dangerous_deserialization=True  # ✅ Required by LangChain 0.2+
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

    # Search SOP vectorstore
    relevant_docs = vectorstore.similarity_search(user_message, k=3)
    kb_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("🔎 Retrieved Context:\n", kb_context)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    {get_prompt()}
                    Use the following internal SOP knowledge base to answer user questions clearly, accurately, and professionally — even if the user does not mention a destination or dates:
                    {kb_context}
                    If you find a match in the context, use it directly in your reply.
                    Do not ignore context. If check-in or pet policies are found, answer them directly.
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