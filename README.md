Ad Optimization with RL and Groq
A smart ad optimization platform leveraging Reinforcement Learning (RL) and Groq API to maximize click-through rates (CTR). Built with a modern, dark-themed Tkinter UI, it offers intelligent ad recommendations and interactive campaign simulations.
Features

Reinforcement Learning: Uses Q-Learning to optimize ad targeting, formats, and display times.
Groq API Integration: Generates compelling ad content and strategies via AI-powered conversations.
Interactive UI: Customize campaigns and simulate performance with real-time results.
Modular Design: Easy to extend with new ad types, audiences, or business types.

Installation

Clone the repository:
git clone https://github.com/Arash-Mansourpour/Reinforcement-Learning_AD-Optimization.git
cd Reinforcement-Learning_AD-Optimization


Install dependencies:Using Poetry (recommended):
poetry install

Or with pip:
pip install -r requirements.txt


Set up Groq API Key:

Obtain a valid Groq API key from xAI API.
Set the key as an environment variable:export GROQ_API_KEY="your_groq_api_key_here"


Alternatively, update src/rlpro.py with your key (not recommended for public repositories):GROQ_API_KEY = "your_groq_api_key_here"





Usage

Run the application:
poetry run python src/rlpro.py

Or:
python src/rlpro.py


Explore the UI:

Campaign Optimization Tab: Select a business type, generate AI-driven recommendations, customize campaigns, and simulate ad performance.
Model Interaction Tab: Chat with the Groq-powered advertising assistant to create ads, get strategy suggestions, or ask questions.


Example Interactions:

Input: "Create an ad for a coffee shop"
Output: JSON with headline, body, and CTA.


Input: "Suggest ad strategies for a SaaS business"
Output: JSON with target audiences, ad types, display times, and goal.


Input: "Hi"
Output: Friendly response asking about your ad needs.





Project Structure
Reinforcement-Learning_AD-Optimization/
├── src/
│   └── rlpro.py           # Main application code
├── .gitignore             # Ignores Python cache and virtual env
├── .github/
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline
├── pyproject.toml         # Poetry configuration
├── requirements.txt       # Dependency list
└── README.md              # Project documentation

Requirements

Python 3.8+
Dependencies: groq==0.9.0, numpy==1.24.0, pandas==2.0.0, customtkinter==5.2.0
A valid Groq API key

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with Groq API for AI-driven ad content.
Powered by Reinforcement Learning for intelligent ad optimization.
UI crafted with CustomTkinter for a modern look.
