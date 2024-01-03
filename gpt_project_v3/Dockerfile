# Utilisez une image de base avec Python
FROM python:3.8

# Créez et définissez le répertoire de travail
WORKDIR /gpt_project_v3

# Copiez le fichier des dépendances dans le conteneur
COPY requirements.txt .

# Installez les dépendances
RUN pip install -r requirements.txt

# Copiez l'ensemble de l'application dans le conteneur
COPY . /gpt_project_v3

# Exposez le port utilisé par Streamlit
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app_blog.py"]