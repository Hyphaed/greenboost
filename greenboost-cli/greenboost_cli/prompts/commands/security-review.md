---
description: Review pending changes for security problems
---
Review the uncommitted changes in this repository (`git diff` and `git diff --staged`) for security problems.
Look for: injection (shell, SQL, path), unsafe deserialisation, secrets in code, missing input validation,
privilege or permission changes, unsafe temp files, and network exposure.
For each finding give: file:line, what is wrong, a concrete exploit sketch, and the smallest fix.
Say plainly when you found nothing. $ARGUMENTS
