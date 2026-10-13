# Transfomer-LM
Un modèle de langage Transformer sur du texte tokenisé en BPE.

Implémentation pédagogique d'un modèle de langage Transformer (1 bloc, 1 tête d'attention, masque causal) entraîné à prédire le token suivant sur du texte français tokenisé en BPE (vocabulaire de 12 000 tokens).

Projet réalisé dans le cadre d'un TP de modèles de langage. Il sert de base pour étudier l'attention, la perplexité et l'effet de la température à la génération.
# Principe
À partir d'une fenêtre de 5 tokens [x1..x5], le modèle lit x = [x1..x4] et doit prédire y = [x2..x5] : chaque position prédit le token qui la suit.
Un masque causal empêche chaque position de lire les positions suivantes. Sans lui, la position 1 verrait la réponse x2 et la perte serait artificiellement nulle.

# Architecture
```text
indices x                        (B, 4)
   │  Embedding 12000→528 + embedding de position 4→528
   ▼                             (B, 4, 528)
   │  Linear 528→256
   ▼                             (B, 4, 256)
   ├──────────────┐
   │  LayerNorm   │
   │  Attention causale, 1 tête
   │  (+) ◄───────┘              connexion résiduelle
   ├──────────────┐
   │  LayerNorm   │
   │  FFN 256→1024→256 (ReLU)
   │  (+) ◄───────┘              connexion résiduelle
   │  LayerNorm finale
   │  Linear 256→12000
   ▼                             (B, 4, 12000)  logits
