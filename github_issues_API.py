# This is code to connect with the Github REST APIs. 
# The code takes repository information as an input and writes the open issues to a text file.
# It also polls the GitHub API to check and retreive new issues created in the repo.

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

#get_contributors(owner, repo)

# Function to retrieve open or closed issues from the project repo as needed
def retrieve_issues(owner, repo):
    open_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "open"})
    #closed_issues = get_github_api(f"repos/{owner}/{repo}/issues", params = {"state": "closed"})
    return open_issues

def store_issues(issues, filename="issues.txt"):
    issue_numbers, issue_titles, issue_URLs, issue_details = [], [], [], []
    if issues:
        for issue in issues:
            issue_numbers.append(issue['number'])
            issue_titles.append(issue['title'])
            issue_URLs.append(issue['html_url'])
            issue_details.append(issue['body'])

        # Write issue data to a text file
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(len(issue_numbers)):
                file.write(f"Issue #{issue_numbers[i]}: {issue_titles[i]}\n")
                file.write(f"  URL: {issue_URLs[i]}\n")
                file.write(f"  Issue details: {issue_details[i]}\n\n")
    else:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("No open issues")
    return issue_numbers, issue_titles, issue_URLs, issue_details


# Variable to store the timestamp of the last check
#last_check_timestamp = None

# Function to poll for new issues since the last check
# def poll_for_new_issues(owner, repo):
#     global last_check_timestamp

#     while True:
#         # Retrieve issues created after the last check timestamp
#         issues = get_github_api(f"repos/{owner}/{repo}/issues",params={"state": "open", "since": last_check_timestamp})

#         if issues:
#             # Process and store new issues
#             issue_numbers, issue_titles, issue_URLs, issue_details = store_issues(issues)
#             store_issues(issue_numbers, issue_titles, issue_URLs, issue_details)

#             # Update the last check timestamp to the current time
#             last_check_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
#         else:
#             print("No new issues.")

#         # Time in between each check
#         time.sleep(600)  # 30 minutes (maybe longer)

#poll_for_new_issues(owner, repo)



issues = retrieve_issues(owner, repo)
store_issues(issues, filename="open_issues.txt")
