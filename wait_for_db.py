import time
import requests

COUCHDB_URL = "http://admin:admin@127.0.0.1:5984/doca_db"
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
    import sys
    sys.exit(1)
    return False

if __name__ == "__main__":
    wait_for_db()
