mkdir -p ~/.streamlit/
echo "[general]
email = \"contact@anhtran.nl\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml