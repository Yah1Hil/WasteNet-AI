# WasteNet AI 🗑️

Real-time waste classification system for Israeli recycling bins.

## Project Overview
- **Author:** Yahav Hilboch  
- **School:** תיכון ע"ש יצחק בן-צבי, קריית אונו
- **Mentor:** Saar Yakulov
- **Subject:** תכנון ותכנות מערכות - 5 יח"ל

## Results
- **Validation Accuracy:** 92.21%
- **Test Accuracy:** 91.89%
- **Dataset:** 16,705 images, 8 categories
- **Architecture:** ResNet18 trained from scratch (no Transfer Learning)

## Key Insight
After 3 runs with ResNet50 (25M params) that maxed at 87.99%, switching to ResNet18 
(11M params) gave a 5% jump in accuracy. **Model size should match dataset size.**

## Categories (Israeli Recycling Bins)
| Color | Category |
|-------|----------|
| Blue | Paper |
| Orange | Plastic & Packaging |
| Purple | Glass |
| Brown | Organic |
| Green | General |
| - | Cardboards |
| - | Electronics |
| - | Clothes |

## Tech Stack
- **Training:** PyTorch, Google Colab Pro (Tesla T4)
- **Application:** Python + PyQt5 + OpenCV + SQLite

## Files
- `run5_clean.py` - Final training script
- `main.py` - Desktop application

## 5 Training Runs
| Run | Model | Val Acc | Note |
|-----|-------|---------|------|
| 1 | ResNet50 + SGD | ~65% | Baseline |
| 2 | ResNet50 + AdamW | 87.99% | Better optimizer |
| 3 | + AMP, LabelSmoothing, Cosine | 86.87% | **Failed - too many techniques** |
| 4 | **ResNet18** | **92.09%** | **Breakthrough** |
| 5 | + MixUp, RandomErasing | 92.21% | Final |# WasteNet AI 🗑️

Real-time waste classification system for Israeli recycling bins.

## Project Overview
- **Author:** Yahav Hilboch  
- **School:** תיכון ע"ש יצחק בן-צבי, קריית אונו
- **Mentor:** Saar Yakulov
- **Subject:** תכנון ותכנות מערכות - 5 יח"ל

## Results
- **Validation Accuracy:** 92.21%
- **Test Accuracy:** 91.89%
- **Dataset:** 16,705 images, 8 categories
- **Architecture:** ResNet18 trained from scratch (no Transfer Learning)

## Key Insight
After 3 runs with ResNet50 (25M params) that maxed at 87.99%, switching to ResNet18 
(11M params) gave a 5% jump in accuracy. **Model size should match dataset size.**

## Categories (Israeli Recycling Bins)
| Color | Category |
|-------|----------|
| Blue | Paper |
| Orange | Plastic & Packaging |
| Purple | Glass |
| Brown | Organic |
| Green | General |
| - | Cardboards |
| - | Electronics |
| - | Clothes |

## Tech Stack
- **Training:** PyTorch, Google Colab Pro (Tesla T4)
- **Application:** Python + PyQt5 + OpenCV + SQLite

## Files
- `run5_clean.py` - Final training script
- `main.py` - Desktop application

## 5 Training Runs
| Run | Model | Val Acc | Note |
|-----|-------|---------|------|
| 1 | ResNet50 + SGD | ~65% | Baseline |
| 2 | ResNet50 + AdamW | 87.99% | Better optimizer |
| 3 | + AMP, LabelSmoothing, Cosine | 86.87% | **Failed - too many techniques** |
| 4 | **ResNet18** | **92.09%** | **Breakthrough** |
| 5 | + MixUp, RandomErasing | 92.21% | Final |
