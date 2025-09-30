from flask import Flask, request, jsonify, render_template, session
from books_recommender import books_data, get_recommendation, genre_match, show_books
from flask_cors import CORS
import os
import pandas as pd
app = Flask(__name__)
app.secret_key = os.urandom(24) # New session every restart (AI use)
CORS(app, resources={r"/*": {"origins": "*"}})
sessions = {}

@app.route("/")
def home():
    session.clear()
    return render_template("chat.html")
@app.route("/chat", methods=["POST"])
def chat():
    print("Full session content:", dict(session))

    try:
        data = request.get_json(force=True)
        user_msg = data.get("message", "").strip()
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    filtered_books = pd.DataFrame()
    step = session.get("step", "choose_preference")
    if "user_input" not in session:
        session["user_input"] = {}

    if step == "choose_preference":
        preference = user_msg.lower()
        if preference not in ["genre", "author"]:
            response = "Invalid choice. Please type 'genre' or 'author'."
        else:
            session["preference"] = preference
            session["step"] = "filter_genre" if preference == "genre" else "filter_author"
            response = "What kind of book genre do you like? If you want to put in 2 or more genres please separate them with a comma." if preference == "genre" else "Who is the author you want to look for?"

    elif step == "filter_genre":
        genres = [g.strip().lower() for g in user_msg.split(",")]
        filtered_books = books_data[books_data['genres'].apply(lambda gs: genre_match(gs, genres))].sort_values('rating_total', ascending=False).head(11)
        filtered_books_copy = filtered_books[['title', 'author']].copy()
        filtered_books_copy['original_index'] = filtered_books_copy.index
        # print(filtered_books)
        if not filtered_books.empty:
            session["filtered_books"] = filtered_books_copy.to_dict("records")
            session["step"] = "select_book"
            response = "Here are the top books:\n"
            for book in session["filtered_books"][:5]:
                response += f"{book['original_index']}: {book['title']} by {book['author']}\n"
            response += "Pick a number or type 'none' to see more."
        else:
            response = "Sorry, no books found. Please enter another genre."

    elif step == "filter_author":
        author_input = user_msg.lower()
        filtered_books = books_data[books_data['author'].str.contains(author_input)].sort_values('rating_total', ascending=False).head(11)
        filtered_books_copy = filtered_books[['title', 'author']].copy()
        filtered_books_copy['original_index'] = filtered_books_copy.index # To avoid errors due to false indexing
        if not filtered_books.empty:
            session["filtered_books"] = filtered_books_copy.to_dict("records")
            session["step"] = "select_book"
            response = "Here are the top books by that author:\n"
            for book in session["filtered_books"][:5]:
                response += f"{book['original_index']}: {book['title']} by {book['author']}\n"
            response += "Pick a number or type 'none' to see more."
        else:
            response = "Sorry, no books found. Please enter another author."

    elif step == "select_book":
        if user_msg.lower() == "none":
            response = "Here are the next 5 books:\n"
            for book in session["filtered_books"][5:10]:
                response += f"{book['original_index']}: {book['title']} by {book['author']}\n"
            response += "Pick a number or type 'keywords' to enter your own keywords."

        elif user_msg.lower() == "keywords":
            session["step"] = "keywords"
            response = "Please enter your keywords for recommendations."
        else:
            try:
                selected_idx = int(user_msg)
                session["user_input"]["selected_idx"] = selected_idx
                valid_indices = [book['original_index'] for book in session["filtered_books"]]

                if selected_idx not in valid_indices:
                    response = "Invalid selection. Please pick a number from the list shown above."
                else:
                    selected_book = next((book for book in session["filtered_books"]
                                          if book.get('original_index') == selected_idx), None)
                    recommended_books = get_recommendation(session["user_input"])
                    response = f"You selected: { selected_book['title' ]}\nHere are your recommended books:\n"
                    for b in recommended_books.to_dict("records"):
                        response += f"- {b['title']} by {b['author']}\n"
                    response += "\nDo you want more recommendations? (y/n)"
                    session["step"] = "more_recommendations"
            except (ValueError, IndexError):
                response = "Invalid selection. Please pick a valid number."

    elif step == "keywords":
        session["user_input"]["keywords"] = user_msg
        try:
            recommended_books = get_recommendation(session["user_input"])
            response = "Here are the recommended books based on your keywords:\n"
            for b in recommended_books.to_dict("records"):
                response += f"- {b['title']} by {b['author']}\n"
            response += "\nDo you want more recommendations? (y/n)"
            session["step"] = "more_recommendations"
        except Exception as e:
            print(f"Recommendation error: {e}")
            response = "Sorry, couldn't generate recommendations. Please try again."

    elif step == "more_recommendations":
        if user_msg.lower() == "y":
            response = "Hi! Do you want recommendations by genre or author?"
            session.clear()
            session["step"] = "choose_preference"
        elif user_msg.lower() == "n":
            response = "Okay, goodbye!"
            session["step"] = "done"
        else:
            response = "Please reply with 'y' or 'n'."

    elif step == "done":
        session.clear()
        response = "Session completed. Refresh to get more recommendations."
    else:
        response = "Session completed. Refresh to get more recommendations."

    return jsonify({"response": response.replace("\n", "<br>")})

if __name__ == "__main__":
    app.run(debug=True)

