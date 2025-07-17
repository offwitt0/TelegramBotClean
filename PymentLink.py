import requests

# API endpoint
url = "https://subscriptionsmanagement-dev.fastautomate.com/api/Payments/reservation"

# Data payload
data = {
   "email": "test4.mohamed@fastautomate.com",
   "amountInCents": 500,
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
