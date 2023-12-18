from flask import Flask, request, jsonify, render_template
import os
import pickle
import joblib
import cv2
import pytesseract
import re
from keras.models import load_model
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load the trained SGDClassifier model
model = joblib.load('contract_classification_model_sgdc.pkl')

# Load the associated label encoder
label_encoder = joblib.load('label_encoder_sgdc.pkl')

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}: consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        return result
    return wrapper
@profile
def image_to_articles(image):
    try:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(threshold_img)

        article_matches = re.finditer(r'\b(?:ARTICLE|Article|article)\b', text)
        indices = [match.start() for match in article_matches]

        # Dictionary to store articles
        articles_dict = {}

        for i in range(len(indices)):
            start_index = indices[i]
            end_index = indices[i + 1] if i + 1 < len(indices) else None
            article_content = text[start_index:end_index].strip()
            if article_content:
                articles_dict[len(articles_dict) + 1] = article_content

        return articles_dict

    except Exception as e:
        return {"error": str(e)}
@profile
def load_suggestion_models(models_dir):
    suggestion_models = {}
    suggestion_tokenizers = {}

    for filename in os.listdir(models_dir):
        if filename.endswith('.h5'):
            model_name = os.path.splitext(filename)[0]

            # Load suggestion model
            suggestion_model_path = os.path.join(models_dir, filename)
            suggestion_model = load_model(suggestion_model_path)
            suggestion_models[model_name] = suggestion_model

            # Load associated tokenizer
            tokenizer_name = f'{model_name.replace("model_", "")}.pickle'
            tokenizer_path = os.path.join(models_dir, tokenizer_name)
            with open(tokenizer_path, 'rb') as handle:
                suggestion_tokenizer = pickle.load(handle)
            suggestion_tokenizers[model_name] = suggestion_tokenizer

    return suggestion_models, suggestion_tokenizers

@profile
def get_suggestions_for_article(article_text, suggestion_model, suggestion_tokenizer, dataset):
    try:
        # Tokenize and pad the new text for content suggestion
        new_text_transformed = suggestion_tokenizer.transform([article_text])

        # Predict the content vector for the new text
        predicted_content_vector = suggestion_model.predict(new_text_transformed)

        # Load the dataset
        df = pd.read_csv(dataset)

        if not df.empty:
            # Tokenize and vectorize the text data (use the existing tokenizer)
            X = suggestion_tokenizer.transform(df['Content'])

            # Predict Content Vectors
            content_vectors = suggestion_model.predict(X)

            # Calculate Content Similarity
            content_similarity = np.dot(content_vectors.astype(np.float32), predicted_content_vector.astype(np.float32).T)
            content_similarity = content_similarity.reshape(-1)

            similarity_threshold = 0.7
            top_content_indices = np.where(content_similarity > similarity_threshold)[0]
            top_content_indices = top_content_indices[top_content_indices != 0]

            if top_content_indices.size > 0:
                # Get recommended content from the DataFrame
                recommended_content = df['Content'].iloc[top_content_indices].tolist()

                # Take the top 5 after filtering
                recommended_content = recommended_content[:5]

                return {'article_content': article_text, 'recommended_content': recommended_content}

        return {'article_content': article_text, 'error': 'No data available for recommendation'}

    except Exception as e:
        print("Error occurred:", e)
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
@profile
def test_mem(articles):
    predicted_class = model.predict([articles[1]])[0]
    predicted_class = label_encoder.inverse_transform([predicted_class])[0]
    return(predicted_class)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@profile
def predict():
    try:
        # Get the uploaded text
        image_file = request.files.get('image')

        # Ensure the filename is a string
        filename = str(image_file.filename)

        # Save the image file to a location
        image_path = os.path.join("uploads", filename)
        image_file.save(image_path)

        # Predict the contract type
        articles = image_to_articles(image_path)
        
        predicted_class = test_mem(articles)

        result_dict = {predicted_class: []}
        print(result_dict)
        # Directory containing suggestion models and tokenizers
        models_dir = 'models_+tokenizer/'

        # Load suggestion models and tokenizers
        suggestion_models, suggestion_tokenizers = load_suggestion_models(models_dir)

        # Load the corresponding suggestion model and tokenizer
        suggestion_model = suggestion_models[predicted_class]
        suggestion_tokenizer = suggestion_tokenizers[predicted_class]

        dataset_mapping = {
            'Contrat de prestation de Service': './contracts/Contrat_de_prestation_de_Service/Contrat_de_prestation_de_Service.csv',
            'cdd': './contracts/cdd/cdd.csv',
            'cdi': './contracts/cdi/cdi.csv',
            'contrat de vente et achat': './contracts/contrat_de_vente_et_achat/contrat de vente et achat.csv',
            'freelance': './contracts/freelance/freelance.csv',
            'location commerciale': './contracts/location commerciale/location commerciale.csv',
            'nda': './contracts/nda/nda.csv'
        }

        for article_number, article_text in articles.items():
            dataset = dataset_mapping[predicted_class]

            result_dict[predicted_class].append(get_suggestions_for_article(article_text,
                                                                           suggestion_model,
                                                                           suggestion_tokenizer,
                                                                           dataset))
        return jsonify(result_dict)

    except Exception as e:
        return jsonify({'error try this out': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
