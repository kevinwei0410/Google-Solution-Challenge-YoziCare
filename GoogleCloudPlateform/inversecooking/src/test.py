import requests

resp = requests.post("https://foodanalysis-z5zukxh7ha-df.a.run.app", files={'file': open('./1.jpg', 'rb')})

print(resp)

print(resp.json())
