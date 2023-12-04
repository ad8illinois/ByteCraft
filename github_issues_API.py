# This is basic code to connect with the Github REST APIs. 
# The code takes repository information as an input and returns project details such as contributors and open issue information.

# TO DO
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

# Get contributors for a project
contributors_data = get_github_api(f"repos/{owner}/{repo}/contributors")
contributors = [contributor["login"] for contributor in contributors_data]

#print("Contributors:", contributors)

# Retreiving open and closed issues from the project repo
open_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "open"})
closed_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "closed"})

issue_numbers, issue_titles, issue_URLs, issue_details,  = [], [], [], []

# Storing issue data in arrays
for issue in open_issues:
    issue_numbers.append(issue['number'])
    issue_titles.append(issue['title'])
    issue_URLs.append(issue['html_url'])
    issue_details.append(issue['body'])

# Function to display issue data 
def display_issues(issue_numbers, issue_titles, issue_URLs, issue_details):
    if issue_numbers:
        for i in range(len(issue_numbers)):
            print(f"Issue #{issue_numbers[i]}: {issue_titles[i]}")
            print(f"  URL: {issue_URLs[i]}")
            print(f"  Issue details: {issue_details[i]}\n")
    else:
        print("No issues to display.")

#display_issues(issue_numbers, issue_titles, issue_URLs, issue_details)

















