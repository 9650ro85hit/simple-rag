# RAG Chatbot (UI prototype)

Node.js + Express + HTML + CSS + vanilla JavaScript. No LLM or database yet:
answers are picked at random from a predefined list, and all data is stored
in the browser's `localStorage`.

## Run

```bash
npm install
npm start
```

Open http://localhost:3000

## Features

- Projects: create, rename (Edit), delete via a three-dot menu. Deleting a project also deletes all its chats.
- Clicking a project expands it and shows its chats; chats can be created inside a project.
- General chatbot: independent chats with rename and delete via a three-dot menu.
- Chat area: messages, question box at the bottom, right-arrow send button (Enter also sends, Shift+Enter adds a new line).
- Random predefined answers (served by `GET /api/answer`, with a local fallback).
- Thumbs-up / thumbs-down on every answer (bottom right); click again to clear.
- Everything persists in `localStorage` (key `ragchat.v1`).

## Structure

```
rag-chatbot/
├── package.json
├── server.js          Express server + /api/answer
└── public/
    ├── index.html
    ├── style.css      light theme
    └── app.js         UI, state, localStorage
```

To reset all data, clear the site's localStorage in your browser dev tools.
