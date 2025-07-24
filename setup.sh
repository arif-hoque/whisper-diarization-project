# Install system dependencies
sudo apt update
sudo apt install -y ffmpeg python3-pip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
if [ ! -f .env ]; then
    echo "# Add your Hugging Face token here" > .env
    echo "HF_TOKEN=\"your_token_here\"" >> .env
    echo "Created .env file - add your Hugging Face token"
fi

echo "Setup complete! Activate virtual environment with: source venv/bin/activate"