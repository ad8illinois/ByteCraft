## Test Data
This folder contains a few issues collected from the numpy github repo: https://github.com/numpy/numpy/issues

The `issues` folder contains some example issues for numpy, organized by the assignee of the issue. For this data-set, we started with the top 5 contributors to numpy, and found issues which were assigned to them.


Issue text was gathered by visitign the webpage, then copy-pasting all the text on the webpage into a file. This means every issues will include some non-relevant data, like the github header and footer text. However, our standard text-retrieval toolkits should be able to handle this data gracefully, using methods like IDF weighting.

## Notes for Real-life applications

- In reality, most issues in numpy don't have any assignee. What we see more often is comments from relevant contributors. We might want to update our logic to get "issues which have been commented on by X contributor" instead of "issues assigned to X contributor"
    - As an example, the 2nd most active contributor "teoliphant" has 0 issues assigned in github.

- Our test data only includes the initial issue description. However, the comments have lots of useful information as well. We might consider also pulling comment text into the data-set.