import requests

owner, repo, branch = "post-no-preference", "options", "master"
query = """SELECT * FROM option_chain WHERE date = '2019-02-09' AND act_symbol = 'A'"""
res = requests.get(
    "https://www.dolthub.com/api/v1alpha1/{}/{}/{}".format(owner, repo, branch),
    params={"q": query},
    headers={ "authorization": "token dhat.v1.gpikpgdn8q7ob73ke8rh2j7j8sq8noiiun69e2cpod4vgnqft1h0" },
)
print(res.json())