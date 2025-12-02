# CI/CD Pipeline Explained - Why Jobs Are Skipped

## 🎯 Why Jobs Are Skipped

The CI/CD pipeline uses **conditional jobs** that only run when a PR is created to a specific branch. This is **correct behavior**!

### Job Conditions

Each job has an `if` condition that determines when it runs:

1. **`code-quality`** → Runs when PR is to `dev`
   ```yaml
   if: github.base_ref == 'dev'
   ```

2. **`model-retraining`** → Runs when PR is to `test`
   ```yaml
   if: github.base_ref == 'test'
   ```

3. **`deploy`** → Runs when PR is to `main` or `master`
   ```yaml
   if: github.base_ref == 'main' || github.base_ref == 'master'
   ```

## 📊 What Happens in Each Scenario

### Scenario 1: PR to `dev` (Feature → dev)
```
PR: feature/my-feature → dev

Jobs that run:
✅ code-quality        (runs because PR is to dev)
⏭️ model-retraining    (skipped - PR is not to test)
⏭️ deploy              (skipped - PR is not to main)
```

### Scenario 2: PR to `test` (dev → test)
```
PR: dev → test (or feature branch → test)

Jobs that run:
⏭️ code-quality        (skipped - PR is not to dev)
✅ model-retraining    (runs because PR is to test)
⏭️ deploy              (skipped - PR is not to main)
```

### Scenario 3: PR to `main` (test → main)
```
PR: test → main (or feature branch → main)

Jobs that run:
⏭️ code-quality        (skipped - PR is not to dev)
⏭️ model-retraining    (skipped - PR is not to test)
✅ deploy              (runs because PR is to main)
```

## ✅ This is Correct!

**The jobs are SUPPOSED to be skipped!** Each job only runs when code is being merged into its target branch:

- Code quality checks run when merging **into dev**
- Model retraining runs when merging **into test**
- Deployment runs when merging **into main**

## 🔄 Complete Workflow Example

### Step 1: Feature → dev
```bash
# Create PR: feature/my-feature → dev
# Only code-quality runs ✅
# After merge, code is in dev
```

### Step 2: dev → test
```bash
# Create PR: dev → test (or create branch from dev and PR to test)
# Only model-retraining runs ✅
# CML report is posted
# After merge, code is in test
```

### Step 3: test → main
```bash
# Create PR: test → main
# Only deploy runs ✅
# Docker image is built and pushed
# After merge, code is in main (production)
```

## 🐛 If You Want All Jobs to Run

If you want to test all jobs in one PR (for testing purposes), you can:

### Option 1: Create PRs to each branch
```bash
# PR 1: feature → dev (runs code-quality)
# PR 2: dev → test (runs model-retraining)
# PR 3: test → main (runs deploy)
```

### Option 2: Modify workflow (not recommended for production)

You could remove the `if` conditions, but this would run all jobs on every PR, which is wasteful and not the intended workflow.

## 📝 Summary

- **Jobs are skipped by design** - they only run for their target branch
- **This is correct behavior** - follows proper CI/CD practices
- **Each stage runs independently** - when code moves to that stage
- **This saves CI/CD resources** - only runs what's needed

Your pipeline is working correctly! 🎉

