# import requests

# class KindoAPI:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://llm.kindo.ai/v1/chat/completions"

#     def call_kindo_api(self, model, messages, max_tokens, **kwargs):
#         headers = {
#             "api-key": self.api_key,
#             "content-type": "application/json",
#         }

#         # Prepare the request payload
#         data = {
#             "model": model,
#             "messages": messages,
#             "max_tokens":max_tokens
#         }

#         # Add optional parameters if any
#         data.update(kwargs)

#         try:
#             # Send the POST request
#             response = requests.post(self.base_url, headers=headers, json=data)
#             response.raise_for_status()
#             return response
#         except requests.exceptions.HTTPError as http_err:
#         # Handle HTTP error responses
#             error_details = {}
#             if response.content:
#                 try:
#                     error_details = response.json()
#                 except requests.exceptions.JSONDecodeError:
#                     error_details = {"error": "Invalid JSON response", "content": response.text}
#             print(f"HTTP error occurred: {http_err}, details: {error_details}")
#             return {"error": str(http_err), "details": error_details}
#         except Exception as err:
#             # Handle other errors (network issues, etc.)
#             print(f"An error occurred: {err}")
#             return {"error": str(err)}


import time
import requests

class KindoAPI:
    MAX_RETRIES = 5  # Number of retries
    INITIAL_WAIT = 10  # Initial wait time in seconds

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://llm.kindo.ai/v1/chat/completions"

    def call_kindo_api(self, model, messages, max_tokens, **kwargs):
        headers = {
            "api-key": self.api_key,
            "content-type": "application/json",
        }

        # Prepare the request payload
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        # Add optional parameters if any
        data.update(kwargs)

        retries = 0
        wait_time = self.INITIAL_WAIT

        while retries < self.MAX_RETRIES:
            try:
                # Send the POST request
                response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
                response.raise_for_status()  # Raise error for 4xx/5xx responses
                return response  # Successful response

            except requests.exceptions.HTTPError as http_err:
                status_code = response.status_code if response else None

                if status_code and 500 <= status_code < 600:
                    print(f"Server error {status_code}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)  # Exponential backoff
                    wait_time *= 2  # Double the wait time for next retry
                    retries += 1
                    continue  # Retry request

                # Handle other HTTP errors (4xx)
                error_details = {"error": "Invalid JSON response", "content": response.text} if response.content else {}
                print(f"HTTP error occurred: {http_err}, details: {error_details}")
                return {"error": str(http_err), "details": error_details}

            except requests.exceptions.RequestException as req_err:
                print(f"Request error occurred: {req_err}")
                return {"error": str(req_err)}

        return {"error": "Max retries exceeded. API unavailable."}
