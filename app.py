from flask import Flask, render_template, request, jsonify



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/recognize', methods=['POST'])
def recognize():
    # Giả lập nhận diện chữ viết tay (thay bằng model AI thực tế)
    result = {"text": "Hello, AI!"}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
    





