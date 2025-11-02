# GitHub Deployment Guide

**Complete Step-by-Step Instructions for Uploading Your Data Science Portfolio**

---

## Prerequisites

Before you begin, ensure you have the following installed on your computer:

- **Git** (version control system)
- **GitHub account** (free account is sufficient)
- **Terminal/Command Prompt** access

### Installing Git

If you don't have Git installed:

**Windows:**
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Open "Git Bash" from the Start menu

**Mac:**
```bash
# Install using Homebrew
brew install git

# Or check if already installed
git --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install git
```

---

## Step 1: Extract Your Portfolio

First, extract the portfolio archive to your desired location:

```bash
# Navigate to where you downloaded the archive
cd ~/Downloads

# Extract the archive
tar -xzf data_science_portfolio.tar.gz

# Navigate into the portfolio directory
cd data_science_portfolio
```

**Windows users:** Right-click the `.tar.gz` file and use 7-Zip or WinRAR to extract.

---

## Step 2: Configure Git (First Time Only)

If this is your first time using Git, configure your identity:

```bash
# Set your name (use your real name)
git config --global user.name "Samwel Munyingi"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## Step 3: Create a GitHub Repository

1. **Go to GitHub:** Navigate to [github.com](https://github.com) and log in

2. **Create New Repository:**
   - Click the **"+"** icon in the top-right corner
   - Select **"New repository"**

3. **Repository Settings:**
   - **Repository name:** `data-science-portfolio` (or your preferred name)
   - **Description:** "Comprehensive data science portfolio showcasing ML, DL, NLP, and analytics projects"
   - **Visibility:** Choose **Public** (recommended for portfolio)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

4. **Click "Create repository"**

5. **Copy the repository URL** (you'll see it on the next page)
   - Example: `https://github.com/yourusername/data-science-portfolio.git`

---

## Step 4: Initialize Local Git Repository

Navigate to your portfolio directory and initialize Git:

```bash
# Make sure you're in the portfolio directory
cd data_science_portfolio

# Initialize Git repository
git init

# Check status (should show untracked files)
git status
```

---

## Step 5: Create .gitignore File

Before committing, create a `.gitignore` file to exclude unnecessary files:

```bash
# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Large model files (optional - remove if you want to include models)
# *.h5
# *.pkl

# Data files (optional - remove if datasets are small)
# *.csv
EOF
```

---

## Step 6: Stage and Commit Files

Add all files to Git and create your first commit:

```bash
# Add all files to staging area
git add .

# Verify what will be committed
git status

# Create your first commit
git commit -m "Initial commit: Complete data science portfolio with 6 projects"

# Verify commit was created
git log --oneline
```

---

## Step 7: Connect to GitHub Remote

Link your local repository to the GitHub repository you created:

```bash
# Add remote repository (replace with YOUR repository URL)
git remote add origin https://github.com/yourusername/data-science-portfolio.git

# Verify remote was added
git remote -v
```

---

## Step 8: Push to GitHub

Upload your portfolio to GitHub:

```bash
# Rename branch to 'main' (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- **Username:** Your GitHub username
- **Password:** Use a **Personal Access Token** (not your account password)

### Creating a Personal Access Token (if needed):

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "Portfolio Upload"
4. Select scopes: Check **"repo"** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)
7. Use this token as your password when pushing

---

## Step 9: Verify Upload

1. **Refresh your GitHub repository page**
2. You should see all your files and folders
3. The README.md should display automatically on the main page

---

## Step 10: Customize Your Repository

### Add Repository Topics (Tags)

1. On your GitHub repository page, click the **⚙️ gear icon** next to "About"
2. Add topics: `data-science`, `machine-learning`, `python`, `portfolio`, `deep-learning`, `nlp`, `analytics`
3. Save changes

### Update Repository Description

Add a compelling description:
```
Comprehensive data science portfolio featuring 6 production-ready projects: ML classification, deep learning, NLP, time series forecasting, and business analytics. Includes working code, visualizations, and interactive dashboards.
```

### Add a Website Link

Link to your portfolio website: `https://samwelmunyingi.com`

---

## Common Issues and Solutions

### Issue 1: "Permission Denied (publickey)"

**Solution:** Use HTTPS instead of SSH, or set up SSH keys:

```bash
# Remove existing remote
git remote remove origin

# Add remote with HTTPS
git remote add origin https://github.com/yourusername/data-science-portfolio.git

# Try pushing again
git push -u origin main
```

### Issue 2: "Repository Not Found"

**Solution:** Check that the repository URL is correct:

```bash
# Check current remote
git remote -v

# Update if incorrect
git remote set-url origin https://github.com/yourusername/data-science-portfolio.git
```

### Issue 3: "Large Files Warning"

**Solution:** If you get warnings about large files (>50MB):

```bash
# Check file sizes
find . -type f -size +50M

# Option 1: Remove large files from tracking
git rm --cached path/to/large/file

# Option 2: Use Git LFS for large files
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS for large model files"
```

### Issue 4: "Authentication Failed"

**Solution:** Use a Personal Access Token instead of password (see Step 8).

### Issue 5: "Merge Conflicts"

**Solution:** If you accidentally initialized the repo with files:

```bash
# Pull and merge
git pull origin main --allow-unrelated-histories

# Resolve any conflicts, then
git push origin main
```

---

## Making Updates After Initial Upload

When you make changes to your portfolio:

```bash
# Check what changed
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Update: Add new visualization to Project 1"

# Push to GitHub
git push origin main
```

---

## Best Practices

### Commit Message Guidelines

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add: New healthcare prediction project"
git commit -m "Fix: Correct typo in README"
git commit -m "Update: Improve model accuracy in churn prediction"
git commit -m "Docs: Add installation instructions"

# Bad examples
git commit -m "changes"
git commit -m "update"
git commit -m "fix stuff"
```

### Regular Commits

Commit frequently with logical groupings:

```bash
# Example workflow
git add project1_customer_churn/
git commit -m "Update: Improve churn prediction model accuracy"

git add project2_sales_dashboard/
git commit -m "Add: New KPI visualizations to sales dashboard"

git push origin main
```

---

## Repository Maintenance

### Keeping Your Repository Clean

```bash
# Remove cached files that shouldn't be tracked
git rm -r --cached __pycache__
git commit -m "Remove: Python cache files"
git push origin main

# Update .gitignore and apply
git rm -r --cached .
git add .
git commit -m "Update: Apply new .gitignore rules"
git push origin main
```

### Creating Branches (Optional)

For experimental features:

```bash
# Create and switch to new branch
git checkout -b feature/new-project

# Make changes, then commit
git add .
git commit -m "Add: New project draft"

# Push branch to GitHub
git push origin feature/new-project

# Merge back to main when ready
git checkout main
git merge feature/new-project
git push origin main
```

---

## Viewing Your Portfolio

Your portfolio is now live at:
```
https://github.com/yourusername/data-science-portfolio
```

Share this link on:
- Your resume
- LinkedIn profile
- Portfolio website
- Job applications

---

## Next Steps

1. ✅ **Star your own repository** (shows engagement)
2. ✅ **Add repository to your GitHub profile README**
3. ✅ **Share on LinkedIn** with a post about your projects
4. ✅ **Pin the repository** to your GitHub profile (up to 6 repos)
5. ✅ **Enable GitHub Pages** (optional) to host project documentation

---

## Quick Reference Commands

```bash
# Clone repository (on a new machine)
git clone https://github.com/yourusername/data-science-portfolio.git

# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# View commit history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- filename
```

---

## Troubleshooting Checklist

Before asking for help, verify:

- ✅ Git is installed: `git --version`
- ✅ You're in the correct directory: `pwd` or `cd`
- ✅ Remote is configured: `git remote -v`
- ✅ You have internet connection
- ✅ GitHub repository exists and URL is correct
- ✅ You're using correct credentials (Personal Access Token)

---

## Support Resources

- **Git Documentation:** [git-scm.com/doc](https://git-scm.com/doc)
- **GitHub Guides:** [guides.github.com](https://guides.github.com)
- **Git Cheat Sheet:** [education.github.com/git-cheat-sheet-education.pdf](https://education.github.com/git-cheat-sheet-education.pdf)

---

**Congratulations!** Your data science portfolio is now live on GitHub and ready to impress potential employers! 🎉
