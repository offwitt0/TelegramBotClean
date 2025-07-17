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
      return print(PaymentUrl)
   else:
      failed = 'failed to send'
      return failed
Payment()