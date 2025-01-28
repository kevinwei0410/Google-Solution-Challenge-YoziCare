import requests

resp = requests.post("https://getprediction-cz6lq3dbtq-de.a.run.app", files={'file': open('eight.png', 'rb')})

print(resp.json())
