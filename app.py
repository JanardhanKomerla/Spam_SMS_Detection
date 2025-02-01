from flask import Flask, request, render_template
import joblib
import json
with open("models/model_metrics.json", "r") as file:
    model_metrics = json.load(file)
tfidf=joblib.load("models/tfidfVectorizer.joblib")
rm_model=joblib.load("models/random_forest_model.pkl")


app=Flask(__name__)

model_metrics = json.load(open("models/model_metrics.json", "r"))

#changing str into float values from json
for key in model_metrics:
    for metric in model_metrics[key]:
        model_metrics[key][metric] = float(model_metrics[key][metric])*100
        model_metrics[key][metric] = round(model_metrics[key][metric],2)


@app.route("/")
def index():
    return render_template("index.html",model_metrics=model_metrics)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "")
    # model_name=request.form.get("model")
    # model_name=model_name.replace(' ', '_').lower()
    # rm_model=joblib.load(f"models/{model_name}_model.pkl")
    # print(f'Recived modelname:{model_name}')
    
    message_tfidf = tfidf.transform([message]).toarray()
    
    # if model_name=='svm':
    #     prediction = rm_model.predict(message_tfidf)[0]
    #     model_name=model_name.replace('_', ' ').capitalize()
    #     return render_template("index.html",model_metrics=model_metrics, prediction=prediction, input_message=message,model_name=model_name)

    # else:
    # prediction = rm_model.predict(message_tfidf)[0]
    probability = rm_model.predict_proba(message_tfidf)[0]
    
    spam_prob = round(probability[1] * 100, 2)  # Probability of Spam
    ham_prob = round(probability[0] * 100, 2) 
    if spam_prob > 40:
        label = "Spam"
    elif 20 <= spam_prob <= 40:
        label = "Might be Spam"
    else:
        label = "Ham"

    
    # model_name=model_name.replace('_', ' ').capitalize()
    return render_template("index.html",model_metrics=model_metrics, prediction=label, input_message=message, spam_prob=spam_prob, ham_prob=ham_prob)


if __name__ == "__main__":
    app.run(debug=True)