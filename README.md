# Titanic Survival Predictor

ML web app predicting passenger survival using Decision Tree.

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```
Visit: `http://localhost:5000`

## Input Fields
| Field | Range | Description |
|-------|-------|-------------|
| Pclass | 1-3 | Ticket class |
| Sex | M/F | Gender |
| Age | 0-100 | Age |
| SibSp | 0-8 | Siblings + Spouse |
| Parch | 0-6 | Parents + Children |
| Fare | 0-600 | Ticket price (£) |
| Embarked | S/C/Q | Port |

## Model
- Algorithm: Decision Tree
- Accuracy: ~78%
- Dataset: Kaggle Titanic (891 passengers)

## Tech Stack
Flask | Scikit-Learn | Pandas | HTML/CSS