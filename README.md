# Face ID Local App

Application desktop Python 100 % locale pour indexer des photos, extraire les visages, comparer avec un référentiel de visages de référence et rechercher les correspondances sans rescanner l'ensemble des images.

## Fonctionnalités visées

- indexation récursive de plusieurs répertoires de photos ;
- reprise après interruption ;
- traitement incrémental des nouvelles photos ;
- stockage SQLite local à l'emplacement choisi par l'utilisateur ;
- indexation incrémentale du répertoire de référence ;
- recherche à partir de l'index existant sans relire toutes les photos ;
- validation, dévalidation et confirmations manuelles persistées ;
- interface graphique PySide6 ;
- test de version du moteur de reconnaissance faciale.

## État actuel

Cette première implémentation pose le socle technique :

- structure du projet Python ;
- persistance des paramètres utilisateur ;
- base SQLite et schéma initial ;
- logique d'inventaire incrémental des photos et références ;
- encapsulation du moteur `InsightFace` ;
- moteur de recherche par similarité sur les embeddings déjà indexés ;
- fenêtre graphique initiale pour piloter le stockage, l'indexation et la recherche ;
- auto-préremplissage au premier lancement avec `MesPhotos/`, `Ref/` et un stockage local `.faceid-data/` si ces dossiers existent dans le workspace ;
- indexation en arrière-plan non bloquante avec barre d'état ;
- bouton d'annulation de l'indexation ;
- bouton **Actualiser l’index** (ne traite que les nouveautés/modifications) ;
- compteur temps réel du nombre de visages détectés ;
- résumé final exporté en UTF-8 dans `last-indexing-summary.json` dans le dossier de stockage.

## Performances et interface

- Recherche NumPy par lots de 256 visages, avec lecture progressive de l’index et récupération groupée des validations.
- Recherche en arrière-plan : la fenêtre reste réactive pendant la comparaison.
- InsightFace limité à la détection et aux embeddings de reconnaissance, sans calcul des attributs âge/genre ni des repères 3D inutilisés.
- Inventaire incrémental sans réécriture des photos et références inchangées ; inventaires volumineux sans limite de paramètres SQLite par fichier.
- Miniatures décodées à taille réduite lorsque le format le permet, orientation EXIF respectée et cache mémoire limité à 128 images.
- Journal limité aux 1 500 dernières lignes pour maîtriser la mémoire.
- Interface claire avec cartes de résultats, chemins abrégés et infobulles, actions activées selon la sélection.
- Pas de vérification réseau des versions au démarrage ni à la fin de l’indexation. Le bouton **Version du moteur** permet la vérification explicite sur PyPI.
- Sauvegarde par fichier conservée pour permettre la reprise après interruption.

Les confirmations sont toujours incluses même sous le seuil ; les rejets sont exclus. Le marquage « manuel » seul conserve l’application du seuil.

## Tests

La suite `pytest` couvre l’indexation, la reprise, les validations, les calculs de similarité par lots, les inventaires et l’interface Qt en mode hors écran. Aucun téléchargement de modèle ni accès réseau n’est nécessaire pour ces tests.

## Lancer le projet

1. Utiliser l'environnement virtuel `.venv` en Python 3.10+.
2. Installer les dépendances avec `requirements.txt`.
3. Dans VS Code, sélectionner l'interpréteur `.venv`.
4. Lancer la configuration **Face ID Local App** ou exécuter `run_app.py` depuis la racine du projet.

## Premier lancement

- si aucun paramètre n'a encore été enregistré, l'application détecte automatiquement :
  - `MesPhotos/` comme dossier photo à indexer ;
  - `Ref/` comme dossier des visages de référence ;
  - `.faceid-data/` comme emplacement de stockage de l'index SQLite ;
- les paramètres sont accessibles via **Paramètres…** (bouton et menu **Outils**) puis sont mémorisés pour les lancements suivants.
- si l'emplacement de stockage n'est pas défini, la boîte de dialogue des paramètres s'ouvre automatiquement au démarrage.
- un mode de performance d'indexation est disponible dans les paramètres (**Rapide / Équilibré / Précis**) pour ajuster vitesse et qualité.

## Données de test du workspace

- Photos : `MesPhotos/`
- Références : `Ref/`

## Remarques

- Le moteur fonctionne entièrement en local.
- Les embeddings sont stockés en base SQLite pour éviter les rescans inutiles.
- Si des répertoires photos configurés se recouvrent (ex: parent + sous-dossier), l'indexation applique une déduplication automatique des fichiers.
- Une mise à jour majeure du moteur pourra nécessiter un rebuild des embeddings ; la version du moteur est donc exposée et testée.
- Le résumé de fin d'indexation (durée, traités, échecs, total visages) est exporté en JSON UTF-8.
