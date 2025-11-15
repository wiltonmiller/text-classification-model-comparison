# CSC311 Group Project — Git Workflow Guide

See below for a clean workflow to avoid conflicst and maintain a clean codebase.

------------------------------------------------------------
1. ALWAYS UPDATE MAIN BEFORE STARTING WORK
------------------------------------------------------------

# Switch to main
git checkout main

# Pull the latest version from GitHub
git pull

------------------------------------------------------------
2. CREATE A NEW FEATURE BRANCH FOR EACH PIECE OF WORK
------------------------------------------------------------

# Create and switch to a branch
git checkout -b feat-<name>

Examples:
git checkout -b feat-preprocessing
git checkout -b feat-model-training
git checkout -b fix-bug-split

All work MUST happen on a feature branch.

------------------------------------------------------------
3. DO YOUR WORK ON THE FEATURE BRANCH
------------------------------------------------------------

# Stage your changes
git add .

# Commit your changes
git commit -m "Message describing what you did"

# Push your branch to GitHub
git push -u origin feat-<name>

------------------------------------------------------------
4. OPEN A PULL REQUEST (PR)
------------------------------------------------------------

Go to GitHub → open a Pull Request from your feature branch into `main`.

Another team member must review your PR before merging.

This avoids conflicts and keeps main clean.

------------------------------------------------------------
5. AFTER THE PR IS MERGED ON GITHUB, UPDATE YOUR LOCAL MACHINE
------------------------------------------------------------

# Switch back to main
git checkout main

# Pull the updated main branch
git pull

------------------------------------------------------------
6. DELETE THE FEATURE BRANCH LOCALLY
------------------------------------------------------------

# Delete the branch since it has been merged
git branch -d feat-<name>

GitHub may also offer to delete the branch online.

------------------------------------------------------------
7. REPEAT FOR THE NEXT FEATURE
------------------------------------------------------------

Every task → new branch → PR → merge → update main.

This ensures:
- No overwriting each other’s work
- No merge-conflict disasters
- A clean repo with clear history
- Easy debugging and tracking

------------------------------------------------------------
Notes
------------------------------------------------------------
- Never push directly to main.
- Never code on main.
- ALWAYS branch → build → PR → merge → update main.
