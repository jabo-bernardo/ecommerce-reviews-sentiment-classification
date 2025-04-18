import joblib

def load_model():
    model = joblib.load("trained/sentiment_model.pkl")
    vectorizer = joblib.load("trained/modelvectorizer.pkl")
    return model, vectorizer

def predict_reviews(reviews, model, vectorizer):
    reviews_tfidf = vectorizer.transform(reviews)
    predictions = model.predict(reviews_tfidf)

    return predictions;


if __name__ == "__main__":
    sample_reviews = [
        "Sobrang ganda ng produkto! Legit na legit. Ang bilis pa ng delivery. Sulit na sulit ang bayad ko. Highly recommended!",
        "Nakakadismaya! Hindi ito yung inaasahan ko. May sira pa pagdating. Sayang lang ang pera ko. Hindi ako oorder ulit dito.",
        "Okay naman yung produkto. Nagana. Pero hindi rin naman ganoon ka-special. Standard lang. Sakto lang sa presyo." ,
        "Ang galing! Tamang-tama sa pangangailangan ko. Matibay at mukhang tatagal. Salamat po sa maayos na serbisyo!",
        "Sobrang tagal dumating. Tapos pagdating, iba pa yung kulay na binigay. Nakakainis! Huwag na kayong umorder dito.",
        "Received the item in good condition.  No issues so far. Will update this review after using it for a while.",
        "Super happy!  Ang cute at ang useful!  Ang dali pang gamitin.  Perfect gift for my friend!",
        "Ang mahal tapos ganito lang pala?  Hindi worth it.  Nag expect ako ng mas maganda.  Disappointed.",
        "Gumagana naman.  Hindi ko pa masyado nasusubukan lahat ng features.  So far, so good.",
        "Best purchase ever! Gumaan ang trabaho ko dahil dito. Ang convenient gamitin. 5 stars!",
        "Never again"
    ]
    
    model, vectorizer = load_model()
    
    predictions = predict_reviews(sample_reviews, model, vectorizer)

    for prediction in zip(sample_reviews, predictions):
        print(prediction[0], '->', prediction[1]);
        print("+++")