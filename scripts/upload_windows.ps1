# Run from project root in Windows PowerShell. Edit URL to exact URL GitHub displays.
# IMPORTANT: Never run until you have reviewed contents, permissions and destination.
$REPO_URL = "https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git"
git init
git branch -M main
git add .
git status
git commit -m "Initial predictive motion-aware 4D Gaussian SLAM research framework"
git remote add origin $REPO_URL
git push -u origin main
