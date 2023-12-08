# This is code to connect with the Github REST APIs. 
# The code takes repository information as an input and writes the issues to a text file.

# Example repos: keras and nx
# https://github.com/keras-team/keras
# https://github.com/nrwl/nx

import requests
import time
import os
import shutil

# GitHub API and repository information
github_api_url = "https://api.github.com"
owner = input("Enter owner name of repository: ") 
repo = input("Enter name of repository: ") 
token = input("Enter your github personal access token: ")

def get_github_api(endpoint, params=None):
    """
    Using the url, owner, repo and token information from above, this function will make a GET request to the GitHub API
    """
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

def get_top_contributors(owner, repo):
    """
    Using repo information, this function will return the top 5 contributors for a project
    """
    contributors_data = get_github_api(f"repos/{owner}/{repo}/contributors")
    sorted_contributors = sorted(contributors_data, key=lambda x: x['contributions'], reverse=True)

    # Get the top 5 contributors
    top_5_contributors = sorted_contributors[:5]
    top_5_contributors = [contributor["login"] for contributor in top_5_contributors]

    return top_5_contributors

def retrieve_issues(owner, repo):
    """
    Retrieves and returns all open issues from the project repo, across all pages 
    """
    issues = []
    page = 1

    while True:
        params = {"state": "open", "page": page}
        response = get_github_api(f"repos/{owner}/{repo}/issues", params=params)

        if not response:
            break  # Exit the loop if there is an issue with the API request

        issues.extend(response)

        # Check if the response is empty or if the number of returned issues is less than the maximum per page
        if not response or len(response) < 30:
            break  # Exit the loop if there are no more pages

        page += 1

    return issues

def store_issues(issues, folder_path="issues"):
    """
    Downloads all the issues from a github repo and save them into a local folder called 'issues'
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # Clear existing files in the folder
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    
    for issue in issues:
        issue_number = issue['number']
        filename = f"{folder_path}/issue_{issue_number}.txt"

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"Issue #{issue_number}: {issue['title']}\n")
            file.write(f"  URL: {issue['html_url']}\n")
            file.write(f"  Issue details: {issue['body']}\n")

def get_issues_commented_by_top_contributors(owner, repo, top_contributors):
    """
    Returns all issues that the top 5 contributors have commented on
    """
    all_issues = retrieve_issues(owner, repo)

    issues_commented_by_top_contributors = []

    for issue in all_issues:
        issue_number = issue['number']
        comments_url = f"repos/{owner}/{repo}/issues/{issue_number}/comments"
        comments_data = get_github_api(comments_url)

        for comment in comments_data:
            commenter_login = comment['user']['login']
            if commenter_login in top_contributors:
                issues_commented_by_top_contributors.append(issue)
                break  # Break once a comment is found by a top contributor for efficiency
    
    store_issues(issues_commented_by_top_contributors, folder_path="issues")
    return issues_commented_by_top_contributors

top_contibutors = get_top_contributors(owner, repo)
print(f"Top contributors: {top_contibutors}")

open_issues = retrieve_issues(owner, repo)
relevant_issues = get_issues_commented_by_top_contributors(owner, repo, top_contibutors)

print(f"Current open issues: {len(open_issues)}")
print(f"Issues that the top 5 contributors have commented on: {len(relevant_issues)}")

# Polling function - commented out since running locally for now

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
