import logging
import requests
from datetime import datetime

user_name = "Test User"
email = "0@example.com"
room_type = "Luxuries Gem"
checkin = datetime(2025, 8, 1)
checkout = datetime(2025, 8, 5)
number_of_guests = 2
amountInCents = 500
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

   print("🔍 Payload to API:", data)  # 👈 Add this

   try:
      response = requests.post(url, json=data)
      print("📨 Status Code:", response.status_code)
      print("📨 Response Text:", response.text)

      if response.status_code == 200:
         session_url = response.json().get("sessionURL")
         print("Stripe Session URL:", session_url)
         return session_url
      else:
         print("Failed request.")
         return None
   except Exception as e:
      logging.error("Payment error: %s", e)
      print("Exception occurred:", e)
      return None

Payment(user_name, email, room_type, checkin, checkout, number_of_guests, amountInCents)
