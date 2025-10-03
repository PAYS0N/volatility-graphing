import requests

owner, repo, branch = "post-no-preference", "options", "master"
query = """
SELECT DISTINCT expiration
FROM option_chain
WHERE act_symbol = 'AAPL';
"""

res = requests.get(
    "https://www.dolthub.com/api/v1alpha1/{}/{}/{}".format(owner, repo, branch),
    params={"q": query},
    headers={ "authorization": "token " },
)
if res.headers.get("Content-Type") == "application/json":
    print(res.json())
else:
    print("Non-JSON response:")
    print("Status:", res.status_code)
    print(res.text[:500])

query = """SELECT expiration 
            FROM option_chain 
            WHERE act_symbol = 'AAPL' 
            AND expiration = '2025-01-29'
            LIMIT 1;"""

query = """SELECT expiration 
        FROM option_chain 
        WHERE act_symbol = 'AAPL' 
        AND expiration = '2025-02-05'
        LIMIT 1;"""
