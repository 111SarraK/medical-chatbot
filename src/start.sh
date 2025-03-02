#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the Streamlit application
exec streamlit run src/chatbot/streamlit_app.py --server.port 8080 --server.address 0.0.0.0