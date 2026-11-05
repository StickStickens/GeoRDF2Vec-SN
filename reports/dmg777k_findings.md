# dmg777k: main outcomes

Full evidence is in [notebooks/dmg777k_eda.ipynb](../notebooks/dmg777k_eda.ipynb).

- **The repo reproduces the paper's graph.** Its parsing gives 776,920 edges and an average degree of 4.6775, both exactly as in the paper. It gives 332,194 nodes, so the paper's 331,194 is probably a typo. The mean edge distance also matches (46.34 km vs 46.37 km), but only with the code's longitude and latitude swapped; the correct order gives 36.50 km. We could not reproduce the mean edge weight (0.763 vs 0.833).
- **Walks are much shorter than the setting suggests.** In the code (the paper never states a direction), they follow outgoing edges only and stop at literals, so with a limit of 8 they average about 1.7 steps. This changes how our adaptive walk length idea should be framed.
- **As committed, the word2vec call trains on nothing,** because a one-shot iterator is used up before training. This must be fixed before any baseline run.
- **Location relates to the class, modestly.** The association is moderate (Cramér's V = 0.30). Coordinates alone do not beat guessing the largest class on accuracy (0.36 at best vs 0.48), but macro F1 doubles compared with shuffled labels (0.26 vs 0.13).
- **Flooding gives many nodes a useless location.** The `Geometry` class is linked to almost every shape, so its inherited location is the middle of the country, and 18.5% of all nodes inherit that same point.
- **Some relations link nearby things, others don't.** Monuments sharing a postal code are typically 30 m from their group's average position, and those sharing a city 370 m. Those sharing a build year or house number are 66–73 km away, about as spread out as monuments in general (75 km). This tells us which relations relation-aware flooding should keep. Monuments already have their own coordinates, so the change can only affect the values around them, such as years and addresses. Whether that improves classification is not yet measured.
