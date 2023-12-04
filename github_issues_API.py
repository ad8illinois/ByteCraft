# This is code to connect with the Github REST APIs. 
# The code takes repository information as an input and returns project details such as contributors and open issue information.
# It also polls the GitHub API to check and retreive new issues created in the repo.

# NOTE: We need administrative access to repos to set up webhooks. We can do this by creating a example repo and testing it but 
# it may be hard for the graders to do this and to show it in the presentation.

# Since we're running this locally, polling should work? We can set it up with whatever intervals we think is good and the function should
# poll the GitHub API to check for new issues and retrieve information about them.


import requests
import time

# GitHub API and repository information
github_api_url = "https://api.github.com"
owner = input("Enter owner name of repository: ") 
repo = input("Enter name of repository: ") 
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
        return None

# Function to get contributors for a project
def get_contributors(owner, repo):
    contributors_data = get_github_api(f"repos/{owner}/{repo}/contributors")
    contributors = [contributor["login"] for contributor in contributors_data]
    return contributors

get_contributors(owner, repo)

# Function to retrieve open or closed issues from the project repo as needed
def retrieve_issues(owner, repo):
    open_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "open"})
    #closed_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "closed"})
    return open_issues

# Function to store issue data in arrays
def store_issues(issues):
    issue_numbers, issue_titles, issue_URLs, issue_details,  = [], [], [], []
    if issues:
        for issue in issues:
            issue_numbers.append(issue['number'])
            issue_titles.append(issue['title'])
            issue_URLs.append(issue['html_url'])
            issue_details.append(issue['body'])
    return issue_numbers, issue_titles, issue_URLs, issue_details

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


# Function to get new issues since a specific date/time
def get_new_issues_since(since):
    endpoint = f"repos/{owner}/{repo}/issues"
    params = {"state": "open", "since": since}
    return get_github_api(endpoint, params)


# Polling function that periodically polls the GitHub API to check for new issues created
def polling():
    # setting to get issues created since this time
    since_timestamp = int(time.time()) - 86400  # Example: 24 hours ago
    while True:
        # Get new issues since the last check
        new_issues = get_new_issues_since(since_timestamp)
        issue_numbers, issue_titles, issue_URLs, issue_details = store_issues(new_issues)
        display_issues(issue_numbers, issue_titles, issue_URLs, issue_details)

        # Update the timestamp for the next check
        since_timestamp = int(time.time())

        # Wait for some time before the next poll (ex. this is set to 10 minutes)
        time.sleep(600)

#polling()
















