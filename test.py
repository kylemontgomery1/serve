import requests

class ClientError(Exception):
    pass

endpoint = 'http://localhost:8080/predictions/vicuna-13b'

response = requests.post(endpoint, json={
    "max_tokens": 128,
    "prompt": "Alan Turing was a ",
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1,
})

try:
    response.raise_for_status()
except Exception as e:
    raise ClientError(
        f"Request failed with {response.status_code}"
    ) from e
result = response.json()
            
print(result)


# import os
# import openai
# openai.api_key = "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"

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "user", "content": "Hello!"}
#   ],
#   max_tokens=5
# )

# print(completion.choices[0].message)
