# Shopee Philippines Tagalog/Taglish Product Reviews Sentiment Classification
This is a sentiment classification model I developed for my university project. It uses a supervised machine learning algorithm called logistic regression, with 15,000 reviews data taken from an open dataset at [HuggingFace](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars) which I modified it for my use case of classification. I added a column for sentiment which I manually labeled. The model has an accuracy of 78%.

## Prerequisites
- Python (>=3.11)
- PIP (>=24.0)

## Installation
```bash
pip install -r requirements.txt
```

## Commands
1. Exploratory Data Analysis
```bash
python eda.py
```
2. Training the model:
```bash
python main.py
```
3. Using the model to classify product reviews (Manually change the array in predict.py file):
```bash
python predict.py
```

4. Hosting the model on a HTTP server
```bash
python server.py
```
Default URL: `http://localhost:5000`
Classification Endpoint: `/predict`
Content-Type: `application/json`
Body:
```json
{
	"content": "Product review goes here"
}
```
Response:
```json
{
	"success": true,
	"response": -1
}
```
`response` could also be 1 which means the given text is a positive review, 0 maps to a neutral review, and lastly, -1 which is a negative review.