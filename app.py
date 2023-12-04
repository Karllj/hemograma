from flask import Flask
from app.routes import app as routes_app  # Importa diretamente o Blueprint

app = Flask(__name__)

# Registrar o Blueprint
app.register_blueprint(routes_app)

if __name__ == '__main__':
    app.run(debug=True)
