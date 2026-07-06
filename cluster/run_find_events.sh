#!/bin/bash
#SBATCH --job-name=find_flood_events
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/events_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/events_%j.err
#SBATCH --cpus-per-task=4              # 4 ליבות מעבד לחישובים מקביליים על האגנים
#SBATCH --mem=24G                       # 24 גיגה זיכרון RAM לטעינת קובצי התוצאות
#SBATCH --time=06:00:00                 # הגבלת זמן רחבה של 6 שעות (יסתיים בהרבה פחות)

# 1. טעינת ה-Conda שהתקנו במעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. אקטיבציה של סביבת הפרויקט
conda activate flashfloods

# 3. פתרון בעיית הדיסק של Matplotlib (במקרה שהסקריפט מייצר הידרוגרפים של האירועים)
export MPLCONFIGDIR=/sci/labs/efratmorin/liron.haris/.matplotlib_cache

# 4. הגדרת משתנה סביבה עבור ה-Home במעבדה
export HOME=/sci/labs/efratmorin/liron.haris/

# 5. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 6. הרצת סקריפט זיהוי האירועים
python src/find_floods_events.py