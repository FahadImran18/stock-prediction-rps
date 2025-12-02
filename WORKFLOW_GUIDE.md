# Git Workflow Guide - How to Work with Branches

## 🎯 Quick Answer

**When you change code, you only work on ONE branch at a time:**

1. **Start on `dev`** - Create a feature branch from `dev`
2. **Work on feature branch** - Make all your changes
3. **Merge via PRs** - Use Pull Requests to move code between branches

**You NEVER manually update multiple branches!** Git handles that through PRs.

## 📋 Complete Workflow

### Scenario 1: Adding a New Feature

```bash
# 1. Start from dev (always start here)
git checkout dev
git pull origin dev

# 2. Create a feature branch
git checkout -b feature/my-new-feature

# 3. Make your changes, commit
git add .
git commit -m "feat: Add new feature"
git push origin feature/my-new-feature

# 4. Create PR: feature → dev
# Go to GitHub, create PR from feature/my-new-feature to dev
# CI will run automatically
# After approval, merge PR

# 5. After merge, delete feature branch locally
git checkout dev
git pull origin dev  # Get the merged changes
git branch -d feature/my-new-feature
```

### Scenario 2: Moving Code from dev → test

```bash
# 1. Make sure dev has all your latest work
git checkout dev
git pull origin dev

# 2. Create a branch for testing
git checkout -b test/ready-for-testing

# 3. Push and create PR: test/ready-for-testing → test
git push origin test/ready-for-testing

# 4. Create PR on GitHub: test/ready-for-testing → test
# CI will run model retraining and CML comparison
# Review CML report
# Merge if model is better

# 5. After merge, test branch now has the code
# No need to manually update anything!
```

### Scenario 3: Moving Code from test → main (Production)

```bash
# 1. Make sure test has the tested code
git checkout test
git pull origin test

# 2. Create a branch for deployment
git checkout -b deploy/to-production

# 3. Push and create PR: deploy/to-production → main
git push origin deploy/to-production

# 4. Create PR on GitHub: deploy/to-production → main
# CD will run, build Docker image, deploy
# Merge after CD passes

# 5. After merge, main now has production code
```

## 🔄 Daily Workflow

### Making Changes

**Always follow this pattern:**

```bash
# 1. Start from dev
git checkout dev
git pull origin dev

# 2. Create feature branch
git checkout -b feature/what-you-are-doing

# 3. Make changes, commit, push
git add .
git commit -m "feat: Description of changes"
git push origin feature/what-you-are-doing

# 4. Create PR on GitHub
# 5. Wait for CI to pass
# 6. Get PR approved and merged
# 7. Pull updated dev
git checkout dev
git pull origin dev
```

## ❌ What NOT to Do

### ❌ DON'T: Manually update multiple branches

```bash
# WRONG - Don't do this!
git checkout dev
# make changes
git commit
git checkout test
git merge dev  # ❌ Don't do this manually
git checkout main
git merge test  # ❌ Don't do this manually
```

### ❌ DON'T: Work directly on dev/test/main

```bash
# WRONG - Don't commit directly to dev/test/main
git checkout dev
# make changes
git commit  # ❌ Use feature branches instead!
```

### ✅ DO: Use Pull Requests

```bash
# CORRECT - Always use PRs
git checkout dev
git checkout -b feature/my-work
# make changes
git commit
git push
# Create PR on GitHub → Let CI/CD handle it
```

## 📊 Branch Status Over Time

```
Day 1:
dev:    [A]──[B]──[C]  (latest features)
test:   [A]──[B]       (tested code)
main:   [A]            (production)

Day 2 (after PRs):
dev:    [A]──[B]──[C]──[D]──[E]  (new features)
test:   [A]──[B]──[C]            (merged from dev)
main:   [A]──[B]                 (merged from test)
```

## 🎯 Key Principles

1. **Always start from `dev`** when creating new work
2. **Use feature branches** for all changes
3. **Use Pull Requests** to move code between branches
4. **Let CI/CD handle** the testing and deployment
5. **Never manually merge** dev→test or test→main (use PRs)

## 🔍 Checking Branch Status

```bash
# See what branch you're on
git branch

# See all branches and their status
git branch -a

# See commit history
git log --oneline --graph --all --decorate -10

# See what's different between branches
git log dev..test  # Commits in dev but not in test
git log test..main # Commits in test but not in main
```

## 🆘 Common Questions

### Q: I made changes on dev, how do I get them to test?

**A:** Create a PR on GitHub:
1. Create a branch from dev: `git checkout -b test/update-from-dev`
2. Push it: `git push origin test/update-from-dev`
3. Create PR: `test/update-from-dev` → `test`
4. Merge after CI passes

### Q: Can I work on multiple features at once?

**A:** Yes! Create separate feature branches:
```bash
git checkout dev
git checkout -b feature/feature-1
# work on feature 1

git checkout dev
git checkout -b feature/feature-2
# work on feature 2
```

### Q: What if I need to fix something in production (main)?

**A:** Create a hotfix branch from main:
```bash
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix
# make fix
git commit
git push
# Create PR: hotfix/critical-fix → main
# After merge, also merge main → dev to keep dev updated
```

## ✅ Summary

- **Work on:** Feature branches (created from `dev`)
- **Move code:** Via Pull Requests on GitHub
- **Never:** Manually update multiple branches
- **Always:** Let CI/CD handle testing and deployment

Your workflow: `feature branch` → PR → `dev` → PR → `test` → PR → `main`

