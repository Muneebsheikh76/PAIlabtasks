from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/joke')
def api_joke():
    try:
        r = requests.get('https://official-joke-api.appspot.com/random_joke', timeout=5)
        r.raise_for_status()
        data = r.json()
        if 'setup' in data and 'punchline' in data:
            text = f"{data.get('setup')} {data.get('punchline')}"
            return jsonify({'joke': text, 'source': 'official-joke-api', 'raw': data})
    except Exception:
        pass
    try:
        r = requests.get('https://v2.jokeapi.dev/joke/Any?type=single', timeout=5)
        r.raise_for_status()
        data = r.json()
        if data.get('joke'):
            return jsonify({'joke': data['joke'], 'source': 'jokeapi.dev', 'raw': data})
    except Exception as e:
        return jsonify({'error': 'Unable to fetch joke', 'details': str(e)}), 502

    return jsonify({'error': 'No joke data found'}), 502


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
