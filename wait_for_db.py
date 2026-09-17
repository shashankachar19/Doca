import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests


def _build_couchdb_url() -> str:
    """Build CouchDB URL from environment variables."""
    base_url = os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984")
    user = os.environ.get("COUCHDB_USER", "")
    password = os.environ.get("COUCHDB_PASSWORD", "")
    db_name = os.environ.get("COUCHDB_DB_NAME", "doca_db")

    if user and password:
        scheme, rest = base_url.split("://", 1)
        return f"{scheme}://{user}:{password}@{rest}/{db_name}"
    return f"{base_url}/{db_name}"


COUCHDB_URL = _build_couchdb_url()
EXPECTED_COUNT = 80
TIMEOUT_SEC = 600

def wait_for_db():
    start = time.time()
    while time.time() - start < TIMEOUT_SEC:
        try:
            resp = requests.get(COUCHDB_URL + "/_all_docs")
            if resp.status_code == 200:
                data = resp.json()
                count = len([row for row in data.get('rows', []) if not row['id'].startswith('_design/')])
                print(f"Current count: {count}")
                if count >= EXPECTED_COUNT:
                    print("All documents processed!")
                    return True
        except Exception as e:
            pass
        time.sleep(5)

    print("Timeout reached!")
    sys.exit(1)
    return False

if __name__ == "__main__":
    wait_for_db()
