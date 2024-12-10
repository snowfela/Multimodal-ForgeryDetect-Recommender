const express = require('express');
const app = express();
const forgeryDetectionRoutes = require('./routes/ForgeryDetectionRoutes');

app.use(express.json());
app.use('/api', forgeryDetectionRoutes);

app.listen(3000, () => console.log('Server running on port 3000'));
