# Text-to-SQL Front End

React interface for the Text-to-SQL Transformer. Takes a question and a list
of table columns, calls the backend API, and displays the generated SQL.

## Setup

```bash
npm install
npm run dev
```

Opens on `http://localhost:5173`. Requires the backend API running on
`http://localhost:8000` - see `../backend/README.md`.

## Structure

```
src/App.jsx    UI and API calls
src/App.css    styles
src/index.css  global reset and CSS variables
```
