from github_issues_API import GithubClient
import multiprocessing
import os


def write_issue(issue, comments, filepath):
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write('----------------\n')
        file.write(f"Issue: #{issue['number']}\n")
        file.write(f"Issue title: {issue['title']}\n")
        file.write(f"Issue url: {issue['html_url']}\n")
        file.write(f"Reporter: {issue['user']['login']}\n")
        file.write('----------------\n')
        file.write(f"{issue['body']}\n")
        file.write('\n\n')
        for comment in comments:
            file.write('----------------\n')
            file.write(f'Comment\n')
            file.write(f"User: {comment['user']['login']}\n")
            file.write('----------------\n')
            file.write(f'{comment["body"]}\n')
            file.write('\n\n')

def process_issue(output_dir, api_token, owner, repo, users, issue):
    """
    Performs the following tasks:
        - Fetches an issue's comments
        - Determines if the issue is relevant to any of the users given
            - If so, stores the issues in a file, and returns it
            - If not, returns None
    
    Implemented as a global function, instead of in a class, so that we can use multiprocessing WorkerPools to run this in parallel
    """
    github = GithubClient(token=api_token, owner=owner, repo=repo)

    issue_number = issue['number']
    reporter = issue['user']['login']

    comments = github.get_comments(issue_number)
    commenters = [c['user']['login'] for c in comments]

    # Determine the relevant user from the reporter and commenters
    issue_users = [reporter, *commenters]
    relevant_user = None
    for user in issue_users:
        if user in users:
            relevant_user = user
    
    if relevant_user is None:
        print(f'Ignoring issue {issue_number}. No relevant users in list {issue_users}')
        return None

    print(f'Saving issue {issue_number}. Relevant user {relevant_user}')
    
    # Save to file
    filepath = os.path.join(output_dir, f'{issue_number}.txt')
    write_issue(issue, comments, filepath)
    
    return {
        'user': relevant_user,
        'filepath': filepath,
    }

class IssueFetcher:
    def __init__(self, owner, repo, token, output_dir):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.output_dir = output_dir
    
    def _github_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }


    def _list_issues(self, limit):
        github = GithubClient(token=self.token, owner=self.owner, repo=self.repo)
        issues = []
        page = 1

        while True:
            res_issues = github.get_issues(page)
            if not res_issues:
                return issues

            for issue in res_issues:
                issues.append(issue)
                if len(issues) >= limit:
                    return issues

            # Number of returned issues is less than the maximum per page, there are no more issues after this
            if len(res_issues) < 30:
                return issues

            page += 1
    
    def fetch_issues_for_users(self, limit, users):
        issues = self._list_issues(limit)
        with multiprocessing.Pool(processes=10) as pool:
            args = [(self.output_dir, self.token, self.owner, self.repo, users, issue) for issue in issues]
            responses = pool.starmap(process_issue, args)
        
        issue_index = {}
        for response in responses:
            if response is None:
                continue

            issue_user = response['user']
            issue_filepath = response['filepath']

            if issue_user not in issue_index:
                issue_index[issue_user] = []
            issue_index[issue_user].append(issue_filepath)
        
        return issue_index

    def fetch_issue(self, issue_number):
        github = GithubClient(token=self.token, owner=self.owner, repo=self.repo)
        issue = github.get_issue(issue_number)
        comments = github.get_comments(issue_number)

        write_issue(issue, comments, os.path.join(self.output_dir, f'{issue["number"]}.txt'))
        return os.path.join(self.output_dir, f'{issue["number"]}.txt')
    