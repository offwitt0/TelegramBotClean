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
import httpx
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for even more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
#hello ive changed this here
#hello im trying to push again
# Payment
async def send_email_to_api(id, email):
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
        print("HTTP Status:", response.status_code)
        result = response.json()  # ✅ Do NOT use `await`
        print("Response JSON:", result)
        return result
        
if __name__ == "__main__":
  asyncio.run(send_email_to_api("123", "test@example.com"))