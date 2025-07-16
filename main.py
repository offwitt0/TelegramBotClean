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
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ================== LOGGING SETUP ==================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
USER_SESSIONS_PATH = "user_sessions.json"

# Load data with error handling
try:
    listings_df = pd.read_excel(EXCEL_PATH)
    logger.info(f"Loaded {len(listings_df)} listings from Excel")
except Exception as e:
    logger.error(f"Error loading Excel file: {e}")
    listings_df = pd.DataFrame()

# Initialize JSON files
for path in [BOOKINGS_PATH, USER_SESSIONS_PATH]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)

# Session management
user_sessions = {}  # session_id -> conversation state
confirmed_sessions = {}  # session_id -> {unit_name, dates}

# ================== ENHANCED BOOKING LOGIC ==================
def is_available(unit_name: str, start_date: datetime.date, end_date: datetime.date) -> bool:
    """Check if a unit is available for the given date range"""
    try:
        with open(BOOKINGS_PATH, "r") as f:
            bookings = json.load(f)
        
        booked_ranges = bookings.get(unit_name, [])
        for booking_range in booked_ranges:
            booked_start = datetime.strptime(booking_range[0], "%Y-%m-%d").date()
            booked_end = datetime.strptime(booking_range[1], "%Y-%m-%d").date()
            
            # Check for overlap
            if not (end_date <= booked_start or start_date >= booked_end):
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return False

def save_booking(unit_name: str, start_date: datetime.date, end_date: datetime.date) -> bool:
    """Save a booking to the JSON file"""
    try:
        with open(BOOKINGS_PATH, "r") as f:
            bookings = json.load(f)
        
        bookings.setdefault(unit_name, []).append([str(start_date), str(end_date)])
        
        with open(BOOKINGS_PATH, "w") as f:
            json.dump(bookings, f, indent=2)
        
        logger.info(f"Booking saved for {unit_name}: {start_date} to {end_date}")
        return True
    except Exception as e:
        logger.error(f"Error saving booking: {e}")
        return False

# ================== ENHANCED NLP & PARSING ==================
def parse_date_from_text(text: str) -> Optional[datetime.date]:
    """Enhanced date parsing from natural language"""
    text = text.lower()
    today = datetime.today().date()
    
    # Specific date patterns
    date_patterns = [
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
        r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})',    # YYYY/MM/DD or YYYY-MM-DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                groups = match.groups()
                if len(groups[2]) == 2:  # 2-digit year
                    year = 2000 + int(groups[2])
                else:
                    year = int(groups[2])
                
                if pattern.startswith(r'(\d{4})'):  # YYYY/MM/DD format
                    return datetime(year, int(groups[1]), int(groups[2])).date()
                else:  # DD/MM/YYYY format
                    return datetime(year, int(groups[1]), int(groups[0])).date()
            except ValueError:
                continue
    
    # Relative date patterns
    if "tomorrow" in text:
        return today + timedelta(days=1)
    elif "next week" in text:
        return today + timedelta(weeks=1)
    elif "next month" in text:
        return today + timedelta(days=30)
    elif "in" in text and "days" in text:
        days_match = re.search(r'in (\d+) days?', text)
        if days_match:
            return today + timedelta(days=int(days_match.group(1)))
    
    return None

def parse_requirements(message: str) -> Dict:
    """Enhanced requirement parsing with better NLP"""
    message = message.lower()
    req = {
        "guests": 0,
        "bathrooms": 0,
        "area": None,
        "pets": False,
        "start_date": None,
        "end_date": None,
        "budget": None,
        "amenities": []
    }
    
    # Area detection (more flexible)
    area_patterns = {
        "zamalek": ["zamalek", "zamalik"],
        "maadi": ["maadi", "maady"],
        "garden city": ["garden city", "garden", "city"],
        "heliopolis": ["heliopolis", "helio"],
        "new cairo": ["new cairo", "cairo"],
        "downtown": ["downtown", "balad"]
    }
    
    for area, patterns in area_patterns.items():
        if any(pattern in message for pattern in patterns):
            req["area"] = area.title()
            break
    
    # Enhanced number parsing
    guest_patterns = [
        r'(\d+)\s*(?:adult|guest|people|person)',
        r'for\s*(\d+)',
        r'(\d+)\s*pax'
    ]
    
    for pattern in guest_patterns:
        match = re.search(pattern, message)
        if match:
            req["guests"] = int(match.group(1))
            break
    
    # Bathroom parsing
    bathroom_match = re.search(r'(\d+)\s*bathroom', message)
    if bathroom_match:
        req["bathrooms"] = int(bathroom_match.group(1))
    
    # Pet detection
    pet_keywords = ["pet", "dog", "cat", "animal"]
    req["pets"] = any(keyword in message for keyword in pet_keywords)
    
    # Date parsing
    req["start_date"] = parse_date_from_text(message)
    if not req["start_date"]:
        req["start_date"] = datetime.today().date() + timedelta(days=7)
    
    # Duration parsing
    duration_match = re.search(r'(\d+)\s*(?:day|night)', message)
    if duration_match:
        req["end_date"] = req["start_date"] + timedelta(days=int(duration_match.group(1)))
    else:
        req["end_date"] = req["start_date"] + timedelta(days=3)
    
    # Budget parsing
    budget_match = re.search(r'(\d+)\s*(?:egp|pound|le)', message)
    if budget_match:
        req["budget"] = int(budget_match.group(1))
    
    # Amenities
    amenity_keywords = {
        "wifi": ["wifi", "internet"],
        "kitchen": ["kitchen", "cooking"],
        "balcony": ["balcony", "terrace"],
        "parking": ["parking", "garage"],
        "pool": ["pool", "swimming"]
    }
    
    for amenity, keywords in amenity_keywords.items():
        if any(keyword in message for keyword in keywords):
            req["amenities"].append(amenity)
    
    return req

def detect_intent(message: str) -> str:
    """Enhanced intent detection"""
    message = message.lower()
    
    # Greeting patterns
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "start"]
    if any(greeting in message for greeting in greetings):
        return "greeting"
    
    # Booking patterns
    booking_keywords = [
        "book", "reserve", "rent", "stay", "apartment", "hotel", "room", 
        "accommodation", "place", "property", "need", "want", "looking for"
    ]
    if any(keyword in message for keyword in booking_keywords):
        return "booking"
    
    # Cancellation patterns
    cancel_keywords = ["cancel", "cancellation", "refund", "change booking"]
    if any(keyword in message for keyword in cancel_keywords):
        return "cancellation"
    
    # Information patterns
    info_keywords = ["help", "information", "about", "services", "amenities", "location"]
    if any(keyword in message for keyword in info_keywords):
        return "information"
    
    # Confirmation patterns
    confirm_keywords = ["yes", "confirm", "book it", "take it", "okay", "ok"]
    if any(keyword in message for keyword in confirm_keywords):
        return "confirmation"
    
    return "question"

# ================== ENHANCED PROPERTY FILTERING ==================
def filter_properties(requirements: Dict) -> pd.DataFrame:
    """Enhanced property filtering with more criteria"""
    if listings_df.empty:
        return pd.DataFrame()
    
    df = listings_df.copy()
    
    # Area filtering
    if requirements.get("area"):
        df = df[df["Area"].str.lower().str.contains(requirements["area"].lower(), na=False)]
    
    # Guest capacity
    if requirements.get("guests") and requirements["guests"] > 0:
        df = df[df["Guests"] >= requirements["guests"]]
    
    # Bathroom count
    if requirements.get("bathrooms") and requirements["bathrooms"] > 0:
        df = df[df["Bathrooms #"] >= requirements["bathrooms"]]
    
    # Pet-friendly (assuming "Luggage" column indicates pet-friendly)
    if requirements.get("pets"):
        df = df[df["Luggage"] == "Yes"]
    
    # Budget filtering (if Price column exists)
    if requirements.get("budget") and "Price" in df.columns:
        df = df[df["Price"] <= requirements["budget"]]
    
    # Availability filtering
    if requirements.get("start_date") and requirements.get("end_date"):
        df = df[df["Unit Name"].apply(
            lambda u: is_available(u, requirements["start_date"], requirements["end_date"])
        )]
    
    # Sort by relevance (you can customize this logic)
    if not df.empty:
        df = df.sort_values(by="Guests", ascending=False)
    
    return df.head(5)  # Return top 5 matches

def generate_airbnb_link(area: str, checkin: datetime.date, checkout: datetime.date, adults: int = 2) -> str:
    """Generate Airbnb search link"""
    area_encoded = quote(f"Cairo, {area}")
    checkin_str = checkin.strftime("%Y-%m-%d")
    checkout_str = checkout.strftime("%Y-%m-%d")
    return f"https://www.airbnb.com/s/{area_encoded}/homes?checkin={checkin_str}&checkout={checkout_str}&adults={adults}"

# ================== ENHANCED RESPONSE GENERATION ==================
def generate_response(user_message: str, session_id: str) -> str:
    """Enhanced response generation with better context awareness"""
    intent = detect_intent(user_message)
    
    # Get or create user session
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "conversation_history": [],
            "current_requirements": {},
            "state": "initial"
        }
    
    session = user_sessions[session_id]
    session["conversation_history"].append({"user": user_message, "timestamp": datetime.now().isoformat()})
    
    if intent == "greeting":
        session["state"] = "greeting"
        return """👋 Hello! Welcome to AnQa Booking Assistant! 

I'm here to help you find the perfect accommodation in Cairo. I can assist you with:

🏠 **Finding Properties** - Tell me your preferences (area, guests, dates, budget)
📅 **Booking Management** - Make reservations and check availability  
ℹ️ **Information** - Answer questions about properties and services
🗺️ **Location Guide** - Help with areas like Zamalek, Maadi, Garden City

How can I help you today? You can say something like:
"I need a 2-bedroom apartment in Zamalek for 4 guests next week" """

    elif intent == "booking":
        session["state"] = "booking"
        requirements = parse_requirements(user_message)
        session["current_requirements"] = requirements
        
        # Store confirmed session data
        confirmed_sessions[session_id] = {
            "candidates": [],
            "requirements": requirements
        }
        
        matching_properties = filter_properties(requirements)
        
        if matching_properties.empty:
            return f"""❌ **No Available Properties Found**

Based on your request:
- Area: {requirements.get('area', 'Any')}
- Guests: {requirements.get('guests', 'Any')}
- Dates: {requirements.get('start_date', 'Not specified')} to {requirements.get('end_date', 'Not specified')}

**Suggestions:**
1. Try a different area or date range
2. Reduce the number of guests
3. Check our available areas: Zamalek, Maadi, Garden City, Heliopolis

Would you like me to help you adjust your search criteria?"""
        
        # Store candidates for potential confirmation
        confirmed_sessions[session_id]["candidates"] = list(matching_properties["Unit Name"])
        
        response_parts = ["🏠 **Available Properties for You:**\n"]
        
        for idx, (_, row) in enumerate(matching_properties.iterrows(), 1):
            unit_name = row["Unit Name"]
            area = row["Area"]
            guests = int(row["Guests"])
            bathrooms = row["Bathrooms #"]
            
            # Generate Airbnb link
            airbnb_link = generate_airbnb_link(
                area, 
                requirements["start_date"], 
                requirements["end_date"], 
                requirements.get("guests", 2)
            )
            
            property_info = f"""**{idx}. {unit_name}** 📍 {area}
👥 Up to {guests} guests | 🚿 {bathrooms} bathrooms
🗓️ Available: {requirements['start_date']} to {requirements['end_date']}
[View on Airbnb]({airbnb_link})"""
            
            response_parts.append(property_info)
        
        response_parts.append(f"\n✅ **To book a property, reply with the exact name** (e.g., '{matching_properties.iloc[0]['Unit Name']}')")
        response_parts.append("❓ **Need more info?** Ask me about amenities, location, or pricing!")
        
        return "\n\n".join(response_parts)
    
    elif intent == "information":
        # Use OpenAI for general information queries
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a helpful hotel and accommodation assistant in Cairo, Egypt. 
                    You help guests with information about:
                    - Cairo neighborhoods (Zamalek, Maadi, Garden City, Heliopolis, etc.)
                    - Local attractions and amenities
                    - Transportation and getting around
                    - Booking policies and procedures
                    - General travel advice for Cairo
                    
                    Be friendly, informative, and helpful. If you don't know something specific about a property, 
                    suggest they ask for more details about specific listings."""},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I apologize, but I'm having trouble processing your question right now. Please try again or ask about specific properties or booking requests."
    
    elif intent == "confirmation":
        # Handle booking confirmation
        confirmation_result = try_confirm_booking(session_id, user_message)
        if confirmation_result:
            session["state"] = "booked"
            return confirmation_result
        else:
            return "I'm not sure which property you'd like to book. Could you please specify the exact property name from the options I provided?"
    
    else:
        # Default to general assistance
        return """I'm here to help you with your accommodation needs in Cairo! 

🔍 **What I can help with:**
- Find and book apartments/hotels
- Answer questions about Cairo areas
- Provide information about amenities
- Help with booking modifications

💡 **Try asking me:**
- "Show me 2-bedroom apartments in Zamalek"
- "What's the best area for tourists?"
- "I need a pet-friendly place for next week"

What would you like to know?"""

def try_confirm_booking(session_id: str, user_message: str) -> Optional[str]:
    """Enhanced booking confirmation with better matching"""
    session_data = confirmed_sessions.get(session_id)
    if not session_data:
        return None
    
    candidates = session_data.get("candidates", [])
    requirements = session_data.get("requirements", {})
    
    # Try to match property name
    user_message_lower = user_message.lower()
    
    for property_name in candidates:
        # Flexible matching - check if property name is contained in message
        if property_name.lower() in user_message_lower:
            # Attempt to save the booking
            if save_booking(
                property_name, 
                requirements.get("start_date"), 
                requirements.get("end_date")
            ):
                # Clear the session
                if session_id in confirmed_sessions:
                    del confirmed_sessions[session_id]
                
                return f"""✅ **Booking Confirmed!**

📋 **Booking Details:**
🏠 Property: **{property_name}**
📅 Check-in: {requirements.get('start_date')}
📅 Check-out: {requirements.get('end_date')}
👥 Guests: {requirements.get('guests', 'Not specified')}

📧 **What's Next:**
- You'll receive a confirmation email shortly
- Check-in instructions will be sent 24 hours before arrival
- For any changes, contact us immediately

🆘 **Need Help?** Reply anytime for assistance!

Thank you for choosing AnQa! 🎉"""
            else:
                return "❌ Sorry, there was an error processing your booking. Please try again or contact support."
    
    return None

# ================== EMAIL HANDLING ==================
def send_email(to_email: str, subject: str, body: str):
    """Enhanced email sending with better formatting"""
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg["Subject"] = f"AnQa Booking Assistant - {subject}"
        
        # Add HTML formatting for better presentation
        html_body = f"""
        <html>
        <body>
            <h2>AnQa Booking Assistant</h2>
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                {body.replace('\n', '<br>')}
            </div>
            <hr>
            <p><i>This is an automated response from AnQa Booking Assistant</i></p>
        </body>
        </html>
        """
        
        msg.set_content(body)  # Plain text version
        msg.add_alternative(html_body, subtype='html')  # HTML version
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            
        logger.info(f"Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Error sending email: {e}")

async def check_email_loop():
    """Enhanced email checking with better error handling"""
    while True:
        try:
            with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
                mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                mail.select("inbox")
                
                # Search for unread emails
                _, messages = mail.search(None, '(UNSEEN)')
                
                if messages[0]:
                    for num in messages[0].split():
                        try:
                            _, msg_data = mail.fetch(num, '(RFC822)')
                            msg = email.message_from_bytes(msg_data[0][1])
                            
                            from_email = email.utils.parseaddr(msg["From"])[1]
                            subject = msg["Subject"] or "No Subject"
                            
                            # Extract body
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                            
                            if body.strip():
                                logger.info(f"📩 Email from {from_email}: {subject}")
                                
                                # Try booking confirmation first
                                confirmation = try_confirm_booking(from_email, body)
                                reply = confirmation or generate_response(body, from_email)
                                
                                # Send reply
                                send_email(from_email, subject, reply)
                                logger.info("✅ Email replied successfully")
                                
                        except Exception as e:
                            logger.error(f"Error processing email: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Email loop error: {e}")
            
        await asyncio.sleep(30)  # Check every 30 seconds

# ================== TELEGRAM HANDLERS ==================
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced start command"""
    welcome_message = """🏨 **Welcome to AnQa Booking Assistant!**

I'm your personal accommodation assistant in Cairo! Here's what I can help you with:

🔍 **Property Search**
- Find apartments and hotels
- Check availability
- Compare options

📅 **Booking Management**  
- Make reservations
- Modify bookings
- Cancel reservations

ℹ️ **Information & Support**
- Area recommendations
- Amenity details
- Local insights

💡 **Get Started:**
Try saying: "I need a 2-bedroom apartment in Zamalek for 4 guests next week"

How can I assist you today?"""
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced message handling"""
    user_message = update.message.text
    user_id = str(update.effective_user.id)
    username = update.effective_user.username or "Unknown"
    
    logger.info(f"📲 Telegram message from @{username} ({user_id}): {user_message}")
    
    try:
        # Try booking confirmation first
        confirmation = try_confirm_booking(user_id, user_message)
        reply = confirmation or generate_response(user_message, user_id)
        
        # Send reply with markdown formatting
        await update.message.reply_text(reply, parse_mode="Markdown", disable_web_page_preview=False)
        
    except Exception as e:
        logger.error(f"Error handling Telegram message: {e}")
        await update.message.reply_text(
            "I apologize, but I encountered an error processing your request. Please try again or contact support.",
            parse_mode="Markdown"
        )

# Add handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ================== FASTAPI APPLICATION ==================
fastapi_app = FastAPI(title="AnQa Booking Assistant API", version="2.0")

@fastapi_app.on_event("startup")
async def startup():
    """Enhanced startup with better logging"""
    logger.info("🚀 Starting AnQa Booking Assistant...")
    
    # Start email monitoring
    logger.info("📧 Starting email monitoring...")
    asyncio.create_task(check_email_loop())
    
    # Initialize Telegram bot
    logger.info("🤖 Initializing Telegram bot...")
    await app.initialize()
    await app.start()
    asyncio.create_task(app.updater.start_polling())
    
    logger.info("✅ AnQa Booking Assistant started successfully!")

@fastapi_app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down AnQa Booking Assistant...")
    await app.stop()
    logger.info("✅ Shutdown complete")

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# API endpoint for direct messaging (optional)
@fastapi_app.post("/chat")
async def chat_endpoint(message: dict):
    """Direct chat API endpoint"""
    user_message = message.get("message", "")
    session_id = message.get("session_id", "api_user")
    
    if not user_message:
        return {"error": "Message is required"}
    
    try:
        confirmation = try_confirm_booking(session_id, user_message)
        reply = confirmation or generate_response(user_message, session_id)
        
        return {"reply": reply, "session_id": session_id}
    except Exception as e:
        logger.error(f"API chat error: {e}")
        return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000, reload=True)