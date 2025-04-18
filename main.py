from train_model import train_model
from predict import predict_reviews
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("reviews.csv")
    
    model, vectorizer = train_model(df)

    sample_reviews = ["Nakakadismaya! Hindi ito yung inaasahan ko. May sira pa pagdating. Sayang lang ang pera ko. Hindi ako oorder ulit dito.", "Okay naman yung produkto. Nagana. Pero hindi rin naman ganoon ka-special. Standard lang. Sakto lang sa presyo.", "What you see is what you get nagmukhang sosyal ang bahay namin, very easy to install it. Lumiwanag na rin ang paligid.", "Ang inorder ko (16w ) (6500k) ang dumating 13w lang.", "Since the addition of this hexagonal honeycomb T5 led tube, the number of people who come to the shop to wash the car has also increased every day, which has become a highlight of my shop! Here I would like to thank the merchant's express delivery, and the reply information is very fast, and will give me a product design match, thank the merchant's effort, I hope this comment can make more users buy"]
    predict_reviews(sample_reviews, model, vectorizer)

    predictions = predict_reviews(sample_reviews, model, vectorizer)

    for prediction in zip(sample_reviews, predictions):
        print(prediction[0], '->', prediction[1]);
        print("+++")
