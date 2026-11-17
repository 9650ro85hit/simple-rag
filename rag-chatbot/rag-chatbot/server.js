
const express = require('express');
const path = require('path');
const { Readable } = require('stream');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

app.use(express.json());

// Forward API requests to the Python FastAPI backend.
app.use('/api', async (req, res) => {
  try {
    const url = `${PYTHON_API_URL}${req.originalUrl}`;

    const headers = { ...req.headers };

    delete headers.host;
    delete headers.connection;
    delete headers['content-length'];
    delete headers['transfer-encoding'];

    const options = {
      method: req.method,
      headers,
    };

    if (!['GET', 'HEAD'].includes(req.method)) {
      if (req.is('application/json')) {
        options.body = JSON.stringify(req.body);
      } else {
        options.body = req;
        options.duplex = 'half';
      }
    }

    const response = await fetch(url, options);

    res.status(response.status);

    response.headers.forEach((value, key) => {
      if (!['content-length', 'connection', 'transfer-encoding'].includes(key.toLowerCase())) {
        res.setHeader(key, value);
      }
    });

    if (!response.body) {
      return res.end();
    }

    Readable.fromWeb(response.body).pipe(res);

  } catch (error) {
    console.error('Python API proxy error:', error);

    if (!res.headersSent) {
      res.status(502).json({
        error: 'Could not connect to Python backend',
      });
    } else {
      res.destroy(error);
    }
  }
});

app.use(express.static(path.join(__dirname, 'public')));

app.listen(PORT, () => {
  console.log(`RAG Chatbot running at http://localhost:${PORT}`);
});
