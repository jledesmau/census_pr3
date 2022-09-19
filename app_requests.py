import requests

url = "https://census-pr3.herokuapp.com/"

response = requests.get(url)  # This should return the response we defined
print(response.status_code)
print(response.json())