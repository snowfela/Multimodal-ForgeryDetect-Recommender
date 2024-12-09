// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
//import './styles/global.css'; // Import the global stylesheet
import App from './App';

// Find the root element
const rootElement = document.getElementById('root');

// Create a root and render the App component
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
