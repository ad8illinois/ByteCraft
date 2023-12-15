# This is code to connect with the Github REST APIs. 
# The code takes repository information as an input and writes the issues to a text file.

import requests

# GitHub API and repository information
github_api_url = "https://api.github.com"

class GithubClient:
    def __init__(self, token: str, owner: str, repo: str):
        self.token = token
        self.owner = owner
        self.repo = repo

    def get_top_contributors(self):
        """
        Using repo information, this function will return the top 5 contributors for a project
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.get(f"https://api.github.com/repos/{self.owner}/{self.repo}/contributors", headers=headers)
        if response.status_code != 200:
            raise Exception(f'Non 200 status code from github api {response.status_code}')
        contributors = response.json()
        sorted_contributors = sorted(contributors, key=lambda x: x['contributions'], reverse=True)

        # Get the top 5 contributors
        top_5_contributors = sorted_contributors[:5]
        top_5_contributors = [contributor["login"] for contributor in top_5_contributors]

        return top_5_contributors
    
    def get_comments(self, issue_number):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.get(f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments", headers=headers)

        if response.status_code != 200:
            print(f'Non 200 status code fetching comments for issue {issue_number}: {response.status_code}')
            return []
        return response.json()
    
    def get_issues(self, page):
        print(f'Fetching issues, page: {page}')
        params = {"state":"all", "page": page, "sort":"created", "direction":"desc"}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(f"https://api.github.com/repos/{self.owner}/{self.repo}/issues", headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f'Non 200 status code from github api {response.status_code}')
        return response.json()
    
    def get_issue(self, issue_number):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.get(f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}", headers=headers)

        if response.status_code != 200:
            raise Exception(f'Non 200 status code from github api {response.status_code}')
        return response.json()