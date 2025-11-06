document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('btn');
  const jokeEl = document.getElementById('joke');
  const sourceEl = document.getElementById('source');

  async function fetchJoke() {
    jokeEl.textContent = 'Loading...';
    sourceEl.textContent = '';
    try {
      const res = await fetch('/api/joke');
      const data = await res.json();
      if (res.ok && data.joke) {
        jokeEl.textContent = data.joke;
        if (data.source) sourceEl.textContent = 'source: ' + data.source;
      } else if (data.error) {
        jokeEl.textContent = 'Error: ' + (data.error || 'Unknown');
        sourceEl.textContent = data.details || '';
      } else {
        jokeEl.textContent = 'Unexpected response';
        sourceEl.textContent = JSON.stringify(data);
      }
    } catch (err) {
      jokeEl.textContent = 'Network error: ' + err.message;
    }
  }

  btn.addEventListener('click', fetchJoke);
});
