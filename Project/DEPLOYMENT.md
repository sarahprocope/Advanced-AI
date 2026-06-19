# Guide de Déploiement Docker de l'API & UI VLM

Ce guide détaille les étapes nécessaires pour dockeriser votre projet de Chatbot VLM et le déployer sur une machine virtuelle (VM) Google Cloud Platform (GCP), en se basant sur la méthodologie du TP8.

---

## 📋 Structure du Projet

Assurez-vous que votre projet respecte la structure suivante (les dossiers `API` et `UI` sont en majuscules dans ce dépôt) :

```text
Advanced-AI/Project/
├── API/
│   ├── Dockerfile
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   ├── pyproject.toml
│   └── uv.lock
├── UI/
│   ├── Dockerfile
│   ├── app.py
│   ├── client.py
│   ├── pyproject.toml
│   └── uv.lock
├── checkpoints/
│   └── best_step1000/       # Vos checkpoints de modèle entraîné
├── docker-compose.yml
└── .gcloudignore             # Optionnel, pour ignorer les fichiers volumineux lors du transfert
```

---

## 1. 🛡️ Configuration des règles de pare-feu GCP

Avant de déployer, vous devez autoriser le trafic entrant sur les ports utilisés par votre application (8000 pour l'API et 7860 pour l'interface Gradio).

Depuis le **terminal de votre machine locale** (et non sur l'instance VM), exécutez les commandes suivantes :

```bash
gcloud compute firewall-rules create allow-vlm-api --allow tcp:8000
gcloud compute firewall-rules create allow-vlm-ui --allow tcp:7860
```

---

## 2. 🚀 Méthode 1 : Déploiement en copiant directement les fichiers sources

Cette méthode copie l'intégralité du code sur la VM et construit les conteneurs directement sur celle-ci.

### Étape 1 : Obtenir le chemin de votre dossier utilisateur sur la VM
Connectez-vous à votre instance et récupérez le chemin de votre dossier personnel :
```bash
pwd
# Retourne généralement quelque chose comme : /home/votre_nom_utilisateur
```

### Étape 2 : Envoyer le projet vers la VM depuis votre machine locale
Dans un terminal sur votre **machine locale**, naviguez à la racine de votre projet et lancez le transfert :
```bash
gcloud compute scp --recurse --zone "us-central1-f" "." "advanced-ai-project":/home/votre_nom_utilisateur/vlm_chatbot
```
> [!IMPORTANT]
> - Remplacez `us-central1-f` par votre zone GCP réelle.
> - Remplacez `advanced-ai-project` par le nom réel de votre instance.
> - Remplacez `votre_nom_utilisateur` par votre identifiant GCP.

> [!TIP]
> Pour accélérer le transfert, vous pouvez créer un fichier `.gcloudignore` à la racine de votre projet pour exclure les environnements virtuels ou caches volumineux :
> ```text
> .git/
> .venv/
> __pycache__/
> .pytest_cache/
> .gradio/
> ```

### Étape 3 : Lancer l'application sur la VM
Connectez-vous en SSH à votre instance, allez dans le dossier et démarrez Docker Compose :
```bash
gcloud compute ssh --zone "us-central1-f" "advanced-ai-project"
cd vlm_chatbot
sudo docker-compose up -d
```
Le drapeau `-d` lance les conteneurs en tâche de fond (detached mode).

---

## 3. 📦 Méthode 2 : Déploiement via des images Docker pré-construites (Recommandé)

Cette méthode consiste à builder les images Docker en local, à les exporter sous forme d'archives compressées, puis à les transférer sur la VM. Cela évite d'utiliser les ressources de la VM pour la compilation et le téléchargement des dépendances.

### Étape 1 : Construire les images Docker localement
Depuis la racine de votre projet sur votre **machine locale**, construisez les images. Nous utilisons `--platform=linux/amd64` pour garantir la compatibilité avec l'architecture Linux standard de la VM (particulièrement important pour les utilisateurs de Mac M1/M2/M3/M4) :

```bash
# Construction de l'image API (contexte de build : racine '.')
sudo docker build --platform=linux/amd64 -f API/Dockerfile -t vlm-chatbot-api:latest .

# Construction de l'image UI (contexte de build : dossier 'UI')
sudo docker build --platform=linux/amd64 -f UI/Dockerfile -t vlm-chatbot-ui:latest UI
```

Vérifiez la création des images :
```bash
sudo docker image ls
# Vous devriez voir vlm-chatbot-api et vlm-chatbot-ui
```

### Étape 2 : Sauvegarder les images dans des fichiers compressés (.tar.gz)
```bash
sudo docker save vlm-chatbot-api:latest | gzip > vlm-chatbot-api.tar.gz
sudo docker save vlm-chatbot-ui:latest | gzip > vlm-chatbot-ui.tar.gz
```

### Étape 3 : Transférer les archives et la configuration sur la VM
Envoyez les images compressées ainsi que le fichier `docker-compose.yml` :
```bash
gcloud compute scp --zone "us-central1-f" vlm-chatbot-api.tar.gz vlm-chatbot-ui.tar.gz advanced-ai-project:/home/votre_nom_utilisateur/
gcloud compute scp --zone "us-central1-f" docker-compose.yml advanced-ai-project:/home/votre_nom_utilisateur/
```

> [!WARNING]
> N'oubliez pas de transférer également votre dossier `checkpoints` s'il n'est pas déjà présent sur la machine virtuelle, car le service API en a besoin pour charger vos poids de modèle entraîné :
> ```bash
> gcloud compute scp --recurse --zone "us-central1-f" checkpoints/ advanced-ai-project:/home/votre_nom_utilisateur/checkpoints/
> ```

### Étape 4 : Mettre à jour `docker-compose.yml` sur la VM
Sur la VM, le fichier `docker-compose.yml` doit être configuré pour utiliser les images Docker pré-construites au lieu de les builder. 

Remplacez les blocs `build` par des directives `image` dans votre `docker-compose.yml` :
```yaml
services:
  api:
    image: vlm-chatbot-api:latest
    ports:
      - "8000:8000"
    volumes:
      - huggingface-cache:/root/.cache/huggingface
      - ./checkpoints:/app/checkpoints
    environment:
      - CHECKPOINT_DIR=/app/checkpoints/best_step1000
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
      interval: 10s
      timeout: 5s
      start_period: 120s
      retries: 5

  ui:
    image: vlm-chatbot-ui:latest
    ports:
      - "7860:7860"
    environment:
      - API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy

volumes:
  huggingface-cache:
```

### Étape 5 : Charger les images et lancer l'application sur la VM
Connectez-vous en SSH à la VM :
```bash
gcloud compute ssh --zone "us-central1-f" "advanced-ai-project"
```
Chargez les images Docker depuis les archives transférées :
```bash
sudo docker load < vlm-chatbot-api.tar.gz
sudo docker load < vlm-chatbot-ui.tar.gz
```
Démarrez les conteneurs :
```bash
sudo docker-compose up -d
```
Vérifiez que tout tourne correctement :
```bash
sudo docker-compose ps
```

---

## 4. 🔗 Accès à l'application

Une fois les conteneurs lancés, récupérez l'**IP externe** de votre instance VM depuis la console Google Cloud. Vous pouvez ensuite accéder à :
- **L'interface Gradio (UI)** : `http://<IP_EXTERNE_VM>:7860`
- **L'API de prédiction** : `http://<IP_EXTERNE_VM>:8000`

---

## ⚙️ Modifier le checkpoint / step du modèle utilisé par l'API

Par défaut, l'API est configurée pour charger le checkpoint situé dans `/app/checkpoints/best_step1000`. Si vous souhaitez utiliser un autre checkpoint (par exemple, suite à un entraînement plus long ou intermédiaire) :

### Étape 1 : S'assurer que le checkpoint est disponible
Le checkpoint doit se trouver dans votre dossier local `checkpoints` (par exemple : `checkpoints/step_2000`).
* Si vous déployez avec la **Méthode 1**, il est copié automatiquement avec le reste du projet.
* Si vous déployez avec la **Méthode 2**, assurez-vous de l'avoir transféré sur la VM :
```bash
gcloud compute scp --recurse --zone "us-central1-f" checkpoints/step_2000 advanced-ai-project:/home/votre_nom_utilisateur/checkpoints/
```

### Étape 2 : Modifier la configuration dans `docker-compose.yml`
Ouvrez le fichier `docker-compose.yml` (en local ou sur la VM selon votre méthode) et modifiez la variable d'environnement `CHECKPOINT_DIR` du service `api` pour cibler votre nouveau dossier (qui est monté dans `/app/checkpoints/`) :

```yaml
  api:
    # ...
    environment:
      - CHECKPOINT_DIR=/app/checkpoints/step_2000  # Modifiez ici le nom du dossier
```

### Étape 3 : Appliquer le changement
Si vous avez modifié le fichier directement sur la VM, relancez simplement la commande suivante :
```bash
sudo docker-compose up -d
```
> [!NOTE]
> Docker Compose détectera le changement de variable d'environnement et recréera automatiquement le conteneur de l'API pour charger le nouveau checkpoint sans perturber le conteneur UI.

---

## 🔄 Mise à jour et Maintenance


Lorsque vous modifiez le code de votre application en local, voici le cycle de mise à jour rapide :

### En local :
1. Reconstruire les images Docker :
   ```bash
   sudo docker build --platform=linux/amd64 -f API/Dockerfile -t vlm-chatbot-api:latest .
   sudo docker build --platform=linux/amd64 -f UI/Dockerfile -t vlm-chatbot-ui:latest UI
   ```
2. Sauvegarder à nouveau sous forme de tar.gz :
   ```bash
   sudo docker save vlm-chatbot-api:latest | gzip > vlm-chatbot-api.tar.gz
   sudo docker save vlm-chatbot-ui:latest | gzip > vlm-chatbot-ui.tar.gz
   ```
3. Transférer sur la VM :
   ```bash
   gcloud compute scp --zone "us-central1-f" vlm-chatbot-api.tar.gz vlm-chatbot-ui.tar.gz advanced-ai-project:/home/votre_nom_utilisateur/
   ```

### Sur la VM :
1. Arrêter l'ancienne version :
   ```bash
   sudo docker-compose down
   ```
2. Charger les nouvelles images Docker :
   ```bash
   sudo docker load < vlm-chatbot-api.tar.gz
   sudo docker load < vlm-chatbot-ui.tar.gz
   ```
3. Relancer l'application :
   ```bash
   sudo docker-compose up -d
   ```
4. Supprimer les fichiers tar temporaires pour libérer de l'espace :
   ```bash
   rm vlm-chatbot-api.tar.gz vlm-chatbot-ui.tar.gz
   ```

---

## 🧹 Nettoyage des ressources

Pour libérer de l'espace sur votre machine locale et sur la VM après le déploiement réussi :
```bash
# Supprimer les archives locales et distantes
rm vlm-chatbot-api.tar.gz vlm-chatbot-ui.tar.gz

# Optionnel: Supprimer les ressources Docker inutilisées sur la VM
sudo docker system prune -a
```

> [!CAUTION]
> **N'oubliez pas d'éteindre votre instance VM sur GCP** lorsque vous n'avez plus besoin du service en ligne afin d'éviter les frais inutiles sur votre crédit académique !
