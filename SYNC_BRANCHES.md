# Branch Synchronization Guide

## Understanding Branch Strategy

**Branches should NOT be the same** - they represent different stages of development:

- **`dev`** → Latest development work (most changes)
- **`test`** → Code ready for testing (should be behind dev, ahead of main)
- **`main`/`master`** → Production-ready code (most stable, behind test)

## Current Branch Status

From your git log:
- ✅ **`dev`**: Has latest changes (flake8 fixes)
- ❌ **`test`**: Way behind (at "Initial" commit)
- ⚠️ **`main`**: Has some changes but different from dev

## Option 1: Sync Branches (Recommended for Starting Fresh)

If you want all branches to start from the same point:

```bash
# Make sure you're on main and it has latest code
git checkout main
git pull origin main

# Update test branch to match main
git checkout test
git merge main
# Or if you want to reset test to match main exactly:
# git reset --hard main
git push origin test

# Update dev branch (dev should have latest, so merge main into dev)
git checkout dev
git merge main
git push origin dev
```

## Option 2: Proper Workflow (Recommended for Production)

Follow the proper branching workflow:

```bash
# 1. Make sure dev has all latest changes
git checkout dev
git pull origin dev

# 2. Merge dev into test (dev → test)
git checkout test
git merge dev
git push origin test

# 3. After testing, merge test into main (test → main)
git checkout main
git merge test
git push origin main
```

## Option 3: Reset All Branches to Same Point

If you want all branches to be identical right now:

```bash
# Get the latest commit (from dev)
git checkout dev
LATEST_COMMIT=$(git rev-parse HEAD)

# Reset test to match dev
git checkout test
git reset --hard $LATEST_COMMIT
git push origin test --force

# Reset main to match dev
git checkout main
git reset --hard $LATEST_COMMIT
git push origin main --force

# Go back to dev
git checkout dev
```

**⚠️ Warning:** Using `--force` will overwrite remote branches. Only do this if you're sure!

## Recommended: Start Fresh with Proper Structure

```bash
# 1. Make dev the source of truth (it has latest changes)
git checkout dev
git pull origin dev

# 2. Update test to be same as dev (for now)
git checkout test
git reset --hard dev
git push origin test --force

# 3. Update main to be same as dev (for now)
git checkout main
git reset --hard dev
git push origin main --force

# 4. Go back to dev for development
git checkout dev
```

## Going Forward: Proper Workflow

After syncing, follow this workflow:

1. **Develop on `dev`**: Make changes, commit, push
2. **Test on `test`**: Create PR from `dev` → `test`, merge after CI passes
3. **Deploy from `main`**: Create PR from `test` → `main`, merge after CD passes

## Which Option Should You Choose?

- **Option 1**: If you want to preserve history but sync branches
- **Option 2**: If you want to follow proper Git workflow (recommended)
- **Option 3**: If you want all branches identical right now (simplest)

**My Recommendation:** Use **Option 3** to sync everything now, then follow **Option 2** going forward.

