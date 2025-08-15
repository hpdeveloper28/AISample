from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from AISample import get_crux_of_lengthy_content_with_output_parser

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    summary = get_crux_of_lengthy_content_with_output_parser(
        "This night is cold in the kingdom I can feel you fade away from the kitchen to the bathroom sink in Your steps keep me awake Don't cut me down throw me out leave me in a waste I once was in man with dignity and grace Now I'm slipping through the cracks of your cold embrace so please please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you want to go, then I'll be so lonely If you leave him, baby let me down slowly If you want to go, then I'll be so lonely If you leave him, baby let me down slowly Cool skin drag my feet on the tile As I'm walking down the corridor We have been talked in a while So I'm looking for an open door Don't cut me down through me I've been in a waste I once was a man with skinny and grace Now I'm slipping through the cracks Should be cold and raised so please Please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down Let me down down Let me down down Let me down If you wanna go then I'll be so lonely If you leave me baby let me down slowly And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down If you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly"
    )
    return render_template("index.html", response_text=summary)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run the Flask app on port 5000
