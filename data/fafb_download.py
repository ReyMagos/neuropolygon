import urllib.request

from root import PROJECT_ROOT


def donwload_asset(name: str):
    API_TOKEN = "<your API token here>"
    DATASET = "fafb"
    API_URL = f"https://codex.flywire.ai/api/download_resource?api_token={API_TOKEN}&data_product={name}&dataset={DATASET}"

    with urllib.request.urlopen(API_URL) as response, open(f"{PROJECT_ROOT}/data", "wb") as file:
        file.write(response.read())
