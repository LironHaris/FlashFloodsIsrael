#!/bin/bash
#SBATCH --job-name=data_type_mask
#SBATCH --output=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/data_type_mask_%j.out
#SBATCH --error=/sci/labs/efratmorin/liron.haris/FlashFloodsIsrael/runs/data_type_mask_%j.err
#SBATCH --cpus-per-task=2              # הסקריפט רץ על ליבה אחת, אחת נוספת לרזרבה
#SBATCH --mem=24G                       # 24 גיגה זיכרון RAM לטעינת hydrographs_unified.csv (כ-15 מיליון שורות)
#SBATCH --time=02:00:00                 # הגבלת זמן של שעתיים (יסתיים בהרבה פחות)

# 1. טעינת ה-Conda שהתקנו במעבדה
source /sci/labs/efratmorin/liron.haris/miniconda3/etc/profile.d/conda.sh

# 2. אקטיבציה של סביבת הפרויקט
conda activate flashfloods

# 3. הגדרת משתנה סביבה עבור ה-Home במעבדה
export HOME=/sci/labs/efratmorin/liron.haris/

# 4. מעבר לתיקיית הפרויקט
cd /sci/labs/efratmorin/liron.haris/FlashFloodsIsrael

# 5. הרצת סקריפט מסכת סוגי הנתונים (flow / flow+normal)
python src/data_type_mask.py
