import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import csv
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        return sia.polarity_scores(review_body)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location:
                valid_locations = [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                    "El Paso, Texas", "Escondido, California", "Fresno, California", "La Mesa, California",
                    "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona",
                    "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]
                if location in valid_locations:
                    filtered_reviews = [review for review in reviews if review['Location'] == location]
                else:
                    filtered_reviews = []

            if start_date and end_date:
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    filtered_reviews = [
                        review for review in filtered_reviews 
                        if start_date <= datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date
                ]
                except ValueError:
                    filtered_reviews = []
            elif start_date:
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                    filtered_reviews = [
                        review for review in filtered_reviews 
                        if start_date <= datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S")
                    ]
                except ValueError:
                    filtered_reviews = []
            elif end_date:
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    filtered_reviews = [
                        review for review in filtered_reviews 
                        if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date
                    ]
                except ValueError:
                    filtered_reviews = []

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        elif environ["REQUEST_METHOD"] == "POST":
            try:
                # Read the POST request body
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length)
                params = parse_qs(request_body.decode('utf-8'))

                # Extract 'ReviewBody' and 'Location'
                review_body = params.get('ReviewBody', [None])[0]
                location = params.get('Location', [None])[0]
                


                missing_fields = [field for field in ['ReviewBody', 'Location'] if not params.get(field, [None])[0]]
                if missing_fields:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Missing required fields: " + ", ".join(missing_fields)}).encode("utf-8")]

                # Validate inputs
                valid_locations = [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                    "El Paso, Texas", "Escondido, California", "Fresno, California", "La Mesa, California",
                    "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona",
                    "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]
                
                if location not in valid_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]
                
                if not review_body or not location:
                    # Added: Error handling for missing fields
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")]

                # Generate ReviewId and Timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Create the new review
                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                # Append the new review to the CSV file
                with open('data/reviews.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=new_review.keys())
                    writer.writerow(new_review)

                # Append the new review to the in-memory list
                reviews.append(new_review)

                # Return the new review as the response
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [("Content-Type", "application/json")])
                return [response_body]

            except Exception as e:
                # Handle any unexpected errors
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

        # If the method is not GET or POST
        start_response("405 Method Not Allowed", [("Content-Type", "application/json")])
        return [json.dumps({"error": "Method not allowed"}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
        