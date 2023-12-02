# This is basic code to connect with the Github REST APIs. 
# The code works to read text on issues that already exist and also returns the contributors on a project. 
# For now, to test if it works, I added inputs for the repo name and issue number but we will need to change this.

# TO DO:
# Need to figure out webhooks and basic automation so we can connect with new issues that are created. (automatic issue management)


import requests
from pprint import pprint

# GitHub API and repository information
github_api_url = "https://api.github.com"
owner = input("Enter owner name of repository: ") # "numpy" for example
repo = input("Enter name of repository: ") # "numpy" for example
token = input("Enter your github personal access token: ")

# Function to make a GET request to GitHub API
def get_github_api(endpoint, params=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(f"{github_api_url}/{endpoint}", headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        pprint(response.json())

# Read issue text
issue_number = input("Enter the issue number: ")  
issue_data = get_github_api(f"repos/{owner}/{repo}/issues/{issue_number}")

print("Issue Title:", issue_data["title"])
print("Issue Body:", issue_data["body"])

# Get contributors for a project
contributors_data = get_github_api(f"repos/{owner}/{repo}/contributors")
contributors = [contributor["login"] for contributor in contributors_data]

print("Contributors:", contributors)

