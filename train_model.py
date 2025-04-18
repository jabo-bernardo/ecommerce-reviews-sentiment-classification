from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data 
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def train_model(df):
    # Parameters already adjusted based on hyperparams
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=7000, min_df=5, max_df=0.7, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    rus = RandomUnderSampler(random_state=42, sampling_strategy = 'auto')
    X_resampled, y_resampled = rus.fit_resample(X_train_tfidf, y_train)

    model = LogisticRegression(solver='liblinear', max_iter=1000, penalty='l2', C=1)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    joblib.dump(model, "trained/sentiment_model.pkl")
    joblib.dump(vectorizer, "trained/modelvectorizer.pkl")

    return model, vectorizer

def train_model_with_tuning(df):
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=5, max_df=0.7)),
        ('rus', RandomUnderSampler(random_state=42, sampling_strategy = 'auto')),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
    ])

    grid_param = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.1, 1, 10, 100] 
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_param,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score ({grid_search.scoring}): {grid_search.best_score_:.4f}")

    print("\nEvaluating the best model on the test set:")
    y_pred = best_estimator.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test Classification Report:\n{classification_report(y_test, y_pred)}")

    return best_estimator, grid_search.best_params_