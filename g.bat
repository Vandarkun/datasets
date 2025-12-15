@echo off
echo [1/3] Adding files...
git add .

echo [2/3] Committing...
git commit -m "%*"

echo [3/3] Pushing...
git push

echo Done!