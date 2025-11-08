@echo off
REM Automated Git Setup Script for Windows
REM This will help you push your project to GitHub

echo =========================================
echo GitHub Setup for DeepLabV3+ Project
echo =========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b
)

echo [OK] Git is installed
echo.

REM Prompt for GitHub username and repository name
set /p GITHUB_USER="Enter your GitHub username: "
set /p REPO_NAME="Enter repository name (default: 18794-HW3-DeepLab): "

if "%REPO_NAME%"=="" set REPO_NAME=18794-HW3-DeepLab

echo.
echo =========================================
echo Configuration:
echo   GitHub User: %GITHUB_USER%
echo   Repository: %REPO_NAME%
echo   URL: https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo =========================================
echo.
echo IMPORTANT: Before running this script, make sure you have:
echo   1. Created the repository on GitHub.com
echo   2. Set it as empty (no README, no .gitignore)
echo.
set /p CONFIRM="Continue? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo =========================================
echo Initializing Git Repository...
echo =========================================

REM Initialize git if not already initialized
if not exist .git (
    git init
    echo [OK] Git repository initialized
) else (
    echo [OK] Git repository already exists
)

echo.
echo =========================================
echo Adding files to Git...
echo =========================================
git add .

echo.
echo =========================================
echo Files to be committed:
echo =========================================
git status

echo.
set /p COMMIT_MSG="Enter commit message (default: Initial commit): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit - DeepLabV3+ implementation

git commit -m "%COMMIT_MSG%"

echo.
echo =========================================
echo Setting up remote repository...
echo =========================================
git remote remove origin 2>nul
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo.
echo =========================================
echo Pushing to GitHub...
echo =========================================
echo You may be asked for your GitHub credentials.
echo Use your GitHub username and Personal Access Token (NOT password)
echo.
echo To create a token: 
echo   GitHub.com ^> Settings ^> Developer settings ^> Personal access tokens
echo.
pause

git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to push to GitHub.
    echo.
    echo Common issues:
    echo 1. Repository doesn't exist - create it on GitHub first
    echo 2. Wrong credentials - use Personal Access Token, not password
    echo 3. Repository not empty - make sure it's created without README
    echo.
) else (
    echo.
    echo =========================================
    echo SUCCESS!
    echo =========================================
    echo Your code is now on GitHub at:
    echo https://github.com/%GITHUB_USER%/%REPO_NAME%
    echo.
)

pause

