##################################################
# Dataloader
# DONE: Erstellung eines Dataloaders für die Daten
# DONE: Cashe auf single ticker basis speichern.
# TODO: Alle ticker herausfinden.
# TODO: Datamanger implementieren. Sollte alle daten laden können. Vorhandene Daten ergänzen können. etc...
# TODO: Survival bias umgehen.

##################################################
# Preprocessor
# DONE: Preprocessing der Daten
# DONE: Berechnung der technischen Indikatoren

###################################################
# Environment
# DONE: Single Agent Portfolio Environment
# DONE: Multi Agent Portfolio Environment mit Shared Observation
# DONE: Multi Agent Portfolio Environment mit Shared Observation und Shared Action
# TODO: Option implementieren für Agenten welche nicht nur den aktuellen State 
# sondern auch die Historie betrachten

####################################################
# Agenten
# DONE: DQN Agent
# DONE: MADDPG Agent
# DONE: Model Builder implementiert
# TODO: Loss function for MADDPG Agent
# TODO: MADDPG Agenten mit TIPP
# TODO: MADDPG Agenten mit CPPI


##################################################
# Generelle TODOs
# DONE: Tensorboard Logging
# DONE: Tensorboard Gewichtungen scheinen nicht zu funktionieren.

# TODO: Datenklassen für reward, state und action implementieren?
# TODO: Visualisierung Stock Preis und Anteil in einem Plot damit man 
# sieht wann der Agent genau kauft bzw verkauft.
# TODO: Hyperparameter Tuning
# TODO: Visualisierungen erstellen.
# TODO: Tracken der besten implementierungen. 

##################################################
# Tests

# DONE: Test für den Dataloader
# DONE: Test für den Preprocessor
# DONE: Test für den Environment
# TODO: Test für den DQN Agent
# DONE: Test für den MADDPG Agent



######################################
# Künftige Erweiterungen

# Andere Agenten:
# TODO: PPO Agent
# TODO: A2C Agent
# TODO: DDPG Agent
# TODO: SAC Agent
# TODO: TD3 Agent
# TODO: DDPG Agent mit TIPP

# TODO: Laden der Modelkonfigurationen aus der config datei. 

# TODO: Ensemble Agenten
# TODO: Ensemble Agenten mit TIPP
# TODO: Ensemble Agenten mit CPPI