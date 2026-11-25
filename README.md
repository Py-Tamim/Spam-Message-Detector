# Spam Message Detector Using Machine Learning 📩

A Python-based Spam Message Detector that classifies text messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Machine Learning.

The project uses **TF-IDF Vectorization** to convert messages into numerical features and **Multinomial Naive Bayes** to classify them.

Users can enter messages directly in the terminal and receive a prediction with a confidence score.

## 📌 Project Overview

Spam messages are commonly used for unwanted advertisements, scams, and fraudulent activities.

This project aims to identify unwanted messages automatically by learning patterns from a labelled SMS dataset.

The system uses:
- **TF-IDF Vectorization:** Converts text messages into numerical features.
- **Multinomial Naive Bayes:** Classifies messages as Spam or Not Spam.

## ✨ Key Features

- Detects spam and legitimate messages.
- Uses TF-IDF for text feature extraction.
- Applies Multinomial Naive Bayes classification.
- Evaluates performance using a classification report.
- Splits data into 80% training and 20% testing.
- Saves the trained model using Joblib.
- Automatically loads an existing model.
- Accepts messages through an interactive terminal interface.
- Displays prediction confidence as a percentage.

## 🤖 Machine Learning Approach

**1. Data Preprocessing**

The dataset contains two columns:

| Column | Description |
|---|---|
| label | Message category: spam or ham |
| text | Actual SMS message |

The labels are converted into numerical values:

- `spam` = 1
- `ham` = 0

**2. TF-IDF Vectorization**

TF-IDF converts message text into numerical representations. The project uses:
- Lowercase text conversion.
- English stop-word removal.
- Unigrams and bigrams (`ngram_range=(1, 2)`).

**3. Multinomial Naive Bayes**

Multinomial Naive Bayes is used to classify messages based on their text features.

The model uses `alpha=0.1` for smoothing.

**4. Model Evaluation**

The dataset is divided into:
- 80% training data.
- 20% testing data.

The `classification_report` evaluates the model using precision, recall, F1-score, and support.

## 🔄 How It Works

1. Loads the SMS Spam Collection dataset.
2. Converts message labels into numerical values.
3. Splits the dataset into training and testing sets.
4. Converts text messages using TF-IDF.
5. Trains the Multinomial Naive Bayes model.
6. Evaluates the model and saves it using Joblib.
7. Accepts new messages from the user.
8. Predicts whether each message is spam or not spam.

## 🖥️ Example Execution

```text
Spam Message Detector - type a message ('quit' to exit)

Message> Congratulations! You have won a free prize.
  -> Spam (95.2% confidence)

Message> Are you coming to class tomorrow?
  -> Not spam (98.1% confidence)

Message> quit
```

*Note: The messages and confidence scores above are illustrative examples, not measured project results.*

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset processing |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Message classification |
| Joblib | Saving and loading the model |

## 📦 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

**2. Install the required libraries**

```bash
python -m pip install scikit-learn pandas joblib
```

**3. Add the dataset**

Place the `SMSSpamCollection` dataset file in the same directory as the Python script.

**4. Run the program**

```bash
python spam_detector.py
```

The model will be trained and saved automatically if no previously trained model exists.

## 📁 Project Structure

```text
Spam-Message-Detector/
│
├── spam_detector.py
├── SMSSpamCollection
├── spam_model.joblib
└── README.md
```
-

## 🤝 AI Assistance & Acknowledgement

I used AI tools as supporting resources during the research and development of this project.

AI assistance included:

- Exploring and refining the initial spam detection project idea.
- Finding suitable CSV and text-based datasets for spam message classification.
- Comparing machine learning algorithms and identifying suitable models for this project.
- Understanding TF-IDF, Naive Bayes, and text classification.

-

## ⚠️ Limitations

- Performance depends on the quality and variety of the training dataset.
- The model mainly learns patterns from message text.
- New spam patterns may not be detected accurately.
- The confidence score is a model probability, not a guarantee that the prediction is correct.

-

## 🚀 Future Improvements

- Test other classification algorithms such as Logistic Regression and Support Vector Machines (SVM).
- Improve text preprocessing.
- Add a graphical user interface.
- Support email spam detection.
- Train with larger and more recent datasets.
- Compare different models using accuracy, precision, recall, and F1-score.


```
** Developed for educational purposes as part of my Python programming practice.

