import argparse
import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ip_address = f"{args.host}:{args.port}"
    api_list = [
        "is_ready",
        "generate",
        "generate_stream",
        "health",
    ]
    for api in api_list:
        try:
            url = f"http://{ip_address}/{api}"
            if api in ["is_ready", "health"]:
                response = requests.get(url)
            else:
                response = requests.post(url)
            response.raise_for_status()
            print(f"api: {api}, response: {response}, response.text: {response.text}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
