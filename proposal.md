## Team Name: ByteCraft

#### Members:
Ben Sivoravong (bs62, bs62@illinois.edu) \
Shivani Mangaleswaran (sm131, sm131@illinois.edu) \
Yogi Patel (ypatel55, ypatel55@illinois.edu) \
Captain: Annamika Dua (ad8, ad8@illinois.edu)
			
#### Topic Proposal:
The role of the version control system (VCS) in today’s software development landscape cannot be understated. VCS platforms like Github and Gitlab serve a variety of key functions in the open source community - code versioning, issue tracking, project documentation, communication, automation, and more. These platforms make it possible for anyone in the world to contribute meaningfully to projects that benefit both themselves and their community, enabling the Open Source ecosystem that we know and love today.

However, the size and scope of many open source projects today creates significant challenges in the area of task management and issue tracking. Taking the popular Python package “numpy” as an example, as of the writing of this report there are 1,983 issues open in its Github repository. Managing this number of issues is a monumental job that would be overwhelming for any single person or team. We have identified 2 common tasks involved with issue management, which may be aided by the use of Text Retrieval algorithms: 
1. Identifying root causes and finding solutions to issues. This requires deep knowledge of the project’s implementation.
2. Identifying duplicate and related issues. This requires shallow knowledge of all the issues in the project.

Text Retrieval can help in both of these tasks:
1. Although text retrieval alone cannot do root-cause analysis and solve problems, it can effectively triage issues to the proper maintainers of the project who have the knowledge required (provided the system is given prior information about the maintainers themselves)
2. Text Retrieval can easily identify duplicate and related issues in a large project, and automatically flag them for review, using basic NLP and document similarity algorithms.

We plan to build a system which accomplishes both of these tasks: automatically triage new issues to project maintainers based on known user profiles of those maintainers, and automatically identify duplicate or high-similarity issues in a project. 

To narrow the scope of this project, we will stick to one VCS platform, either Github or Gitlab. 

#### Programming Language: 
We plan on using Python for our project, because of the availability of libraries common in NLP and TR (like numpy, metapy, etc).

#### Workload Justification:
Please justify that the workload of your topic is at least 20*N hours, N being the total number of students in your team. You may list the main tasks to be completed, and the estimated time cost for each task.

| Main Tasks      | Estimated Time Cost (hours) |
| ----------- | ----------- |
| Basic automation - Trigger actions when new issues are created, interact with VCS API’s for automatic issue management.  | 20 hours       |
| Tokenization, LM initialization - read project source code, issues to instantiate bag-of-words vector dimensions. Then transform issue descriptions into vectors.   | 20 hours        |
| User Profiles - Use git commit history and user-provided data to assign bag-of-words vectors to each maintainer of the project.      | 20 hours       |
| Similarity and Ranking - Implement ranking functions to generate a ranked list of maintainers for each Issue.   | 10 hours        |
| Integration and polish - Bring all these components together into a unified system that can actually be used by people.      | 10 hours       |
| (Stretch goal) Duplicate, similarity detection - use similarity between queries to automatically label duplicate or related issues.   | 10 hours        |
| **Total**      | **90 hours (10 stretch)**       |
