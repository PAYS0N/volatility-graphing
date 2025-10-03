import subprocess
import json

# SQL query
query = "SELECT date, strike, call_put, bid, ask, vol FROM option_chain WHERE act_symbol='AAPL' AND expiration BETWEEN '2025-01-29' and '2025-02-05' AND date >= '2025-01-20'"

# Run the query using dolt sql
print("Querying options for expiration date in week range")
result = subprocess.run(
    ["dolt", "sql", "-q", query, "-r", "json"],
    capture_output=True,
    text=True,
    cwd="./options"  # path to Dolt repo
)

if result.returncode != 0:
    print("Dolt failed:", result.stderr)
    raise RuntimeError("Query failed")

if not result.stdout.strip():
    raise ValueError("No output from Dolt query")

print(result.stdout)
# print("Formatting output")
# # Parse JSON
# data = json.loads(result.stdout)
# # data['rows'] is a list of lists
# expiration = data['rows'][0]['expiration']
# print(expiration)