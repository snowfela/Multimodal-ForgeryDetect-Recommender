const spawn = require('child_process').spawn;

exports.detectForgery = async (req, res) => {
    const imagePath = req.file.path; // Uploaded file path
    const process = spawn('python3', ['./src/backend/fd_gan/main.py', imagePath]);
    
    process.stdout.on('data', (data) => {
        const result = data.toString();
        res.status(200).json({ message: 'Forgery detection completed', result });
    });

    process.stderr.on('data', (error) => {
        res.status(500).json({ error: error.toString() });
    });
};
