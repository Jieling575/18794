# How to Upload This Project to GitHub

This guide shows you how to upload your DeepLabV3+ project to GitHub so you can easily access it from AWS or anywhere else.

## 🎯 Why Use GitHub?

- ✅ Easy to clone on AWS (just `git clone`)
- ✅ Version control for your code
- ✅ Backup your work
- ✅ Share with collaborators
- ✅ No need to manually upload files via SCP

---

## Method 1: Using GitHub Desktop (Easiest for Windows)

### Step 1: Install GitHub Desktop
1. Download from: https://desktop.github.com/
2. Install and sign in with your GitHub account
3. If you don't have a GitHub account, create one at https://github.com

### Step 2: Create Repository
1. Open GitHub Desktop
2. Click **File** → **New Repository**
3. Fill in:
   - **Name**: `18794-HW3-DeepLab`
   - **Description**: `DeepLabV3+ semantic segmentation on Pascal VOC`
   - **Local Path**: `C:\Users\HP\Desktop\18794_HW3_F25`
   - **Initialize with README**: Uncheck (you already have files)
   - **.gitignore**: None (we already created one)
4. Click **Create Repository**

### Step 3: Add Your Files
1. GitHub Desktop should show all your files in the left panel
2. Check the boxes for files you want to include
3. At the bottom:
   - **Summary**: `Initial commit - DeepLabV3+ implementation`
   - **Description**: (optional) `Complete implementation with ResNet50 backbone`
4. Click **Commit to main**

### Step 4: Publish to GitHub
1. Click **Publish repository** (top right)
2. Uncheck **Keep this code private** (or keep it checked if you want it private)
3. Click **Publish Repository**

### Step 5: Done!
Your repository is now at: `https://github.com/YOUR_USERNAME/18794-HW3-DeepLab`

---

## Method 2: Using Command Line (Git Bash or PowerShell)

### Step 1: Install Git
If you don't have Git installed:
1. Download from: https://git-scm.com/download/win
2. Install with default settings

### Step 2: Create GitHub Repository
1. Go to https://github.com
2. Click the **+** icon (top right) → **New repository**
3. Fill in:
   - **Repository name**: `18794-HW3-DeepLab`
   - **Description**: `DeepLabV3+ semantic segmentation on Pascal VOC`
   - **Public** or **Private**: Your choice
   - **DO NOT** check "Initialize with README" (you have existing code)
4. Click **Create repository**
5. Keep this page open (you'll need the URL)

### Step 3: Push Your Code

Open **PowerShell** or **Git Bash**:

```bash
# Navigate to your project
cd C:\Users\HP\Desktop\18794_HW3_F25\hw3_semantic_segmentation

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Commit your files
git commit -m "Initial commit - DeepLabV3+ implementation"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/18794-HW3-DeepLab.git

# Push to GitHub (you may be asked for GitHub credentials)
git push -u origin main
```

If you get an error about branch name, try:
```bash
git branch -M main
git push -u origin main
```

### Step 4: Enter GitHub Credentials
- Windows will prompt for GitHub credentials
- Use your GitHub username and **Personal Access Token** (not password)
- To create a token: GitHub → Settings → Developer settings → Personal access tokens → Generate new token

---

## 🚀 Using Your GitHub Repository on AWS

Once your code is on GitHub, using it on AWS is super easy:

### On AWS Instance:

```bash
# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/18794-HW3-DeepLab.git

# Navigate to project
cd 18794-HW3-DeepLab

# Activate PyTorch environment
source activate pytorch_p38

# Install requirements
pip install -r requirements.txt

# Download dataset
cd datasets
mkdir -p data
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
mkdir -p VOC2012_train_val
mv VOCdevkit/VOC2012 VOC2012_train_val/VOC2012_train_val

# Go back and start training
cd ~/18794-HW3-DeepLab
python main.py --model deeplabv3plus_resnet50 --output_stride 16 --batch_size 8 --lr 0.01 --lr_policy step --step_size 1000 --total_itrs 5000 --gpu_id 0
```

---

## 📝 What Gets Uploaded to GitHub?

✅ **Included** (source code):
- `main.py`
- `plot_miou.py`
- `network/` folder
- `datasets/` folder (only .py files, NOT data)
- `metrics/` folder
- `utils/` folder
- `requirements.txt`
- `README.md`

❌ **Excluded** (too large or temporary):
- `datasets/data/` (dataset is huge, download separately)
- `checkpoints/` (model files are large)
- `__pycache__/` (Python cache)
- `*.log` files
- `*.npy` files
- Result images

This keeps your repository small and fast!

---

## 🔄 Updating Your Code on GitHub

After making changes:

### Using GitHub Desktop:
1. Open GitHub Desktop
2. It shows all changes automatically
3. Write commit message
4. Click **Commit to main**
5. Click **Push origin**

### Using Command Line:
```bash
cd C:\Users\HP\Desktop\18794_HW3_F25\hw3_semantic_segmentation

# See what changed
git status

# Add all changes
git add .

# Commit with message
git commit -m "Updated training parameters"

# Push to GitHub
git push
```

---

## 🔐 Making Repository Private

If you want to keep your code private (recommended for homework):

1. Go to your repository on GitHub
2. Click **Settings** (tab at top)
3. Scroll to bottom → **Danger Zone**
4. Click **Change repository visibility**
5. Select **Make private**

To give your instructor access:
1. Go to **Settings** → **Collaborators**
2. Click **Add people**
3. Enter instructor's GitHub username

---

## 🛠️ Troubleshooting

### Problem: "git: command not found"
**Solution**: Install Git from https://git-scm.com/download/win

### Problem: "Permission denied" when pushing
**Solution**: You need a Personal Access Token:
1. GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic) → Generate new token
3. Select scopes: `repo` (all)
4. Generate and copy token
5. Use this token instead of password when prompted

### Problem: Files are too large (>100MB)
**Solution**: Make sure `.gitignore` is working
```bash
# Check if .gitignore exists
ls -la .gitignore

# If dataset files are tracked, remove them
git rm -r --cached datasets/data/
git commit -m "Remove dataset from tracking"
```

### Problem: Want to ignore files already committed
**Solution**:
```bash
# Remove from git but keep locally
git rm --cached filename

# Or for folders
git rm -r --cached foldername/

# Commit the change
git commit -m "Remove tracked files"
```

---

## 📦 Quick Reference

```bash
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push

# Pull latest changes
git pull

# Clone repository
git clone https://github.com/USERNAME/REPO.git

# View remote URL
git remote -v
```

---

## ✅ Success Checklist

- [ ] .gitignore file created
- [ ] Git initialized in your project
- [ ] All source code committed
- [ ] Repository created on GitHub
- [ ] Code pushed to GitHub successfully
- [ ] Can view your code at github.com/YOUR_USERNAME/REPO
- [ ] Dataset files NOT uploaded (too large)
- [ ] Repository set to private (if desired)

---

## 🎓 For Your Homework

**Important Notes:**
1. **Keep repository private** if this is for a class (academic integrity)
2. **Don't upload datasets** - they're too large and available publicly
3. **Don't upload checkpoints** - regenerate them during training
4. **DO upload** all your source code and README

Your instructor can access your private repo if you add them as a collaborator.

---

Good luck! 🚀

