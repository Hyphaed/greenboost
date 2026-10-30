---
description: Summarise the review comments on the current pull request
---
Run `gh pr view --comments` (and `gh api repos/{owner}/{repo}/pulls/{number}/comments` if inline comments are missing) for the
current branch's pull request. Group the comments by file, say which are already addressed in the current code, and list what is
still open with the smallest change that would resolve each. $ARGUMENTS
