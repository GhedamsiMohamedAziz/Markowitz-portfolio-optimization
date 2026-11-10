# Markowitz Portfolio Optimization (improved)

Optimisation moyenne-variance (Markowitz) sur un univers d'actions US : frontière efficiente, portefeuille max Sharpe et portefeuille volatilité minimale, avec contrainte long-only.

## Structure

- `data_loader.py` : téléchargement des prix via `yfinance`, cache CSV local, calcul des rendements.
- `optimizer.py` : statistiques annualisées, nuage de portefeuilles aléatoires (seed fixe pour la reproductibilité), optimisation SLSQP (max Sharpe / min vol / frontière).
- `plots.py` : graphiques (frontière + nuage coloré par Sharpe, allocations).
- `main.py` : exécution complète, affiche les allocations et sauvegarde les PNG.

## Utilisation

```bash
pip install -r requirements.txt
python main.py
```

Sortie : allocations des portefeuilles optimaux dans le terminal + `efficient_frontier.png` et `max_sharpe_weights.png`.

## Améliorations par rapport au notebook initial

- Code modularisé et testable au lieu d'un notebook monolithique.
- Cache des données pour éviter les téléchargements répétés.
- Seed fixe : résultats reproductibles.
- Contrainte long-only explicite, gestion des cibles inatteignables sur la frontière.
- Erreurs d'optimisation propagées proprement au lieu d'être silencieuses.
- Pas de dépendance au fetch manuel de données : un `python main.py` suffit.

`Markowitz.ipynb` reste dans le repo comme version d'exploration d'origine.
