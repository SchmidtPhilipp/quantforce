##################################################
# Wichtige TODOs
# DONE: Asset Tracker für beliebige anzahl an Episoden implementieren.
# DONE: Implementiere einen generallen Tracker.
# DONE: Überprüfe den Dataloader.
# TODO: Visualize torch gradients
# TODO: optimize the code for MPS
# TODO: Achsenbeschriftung im Tracker mitgeben.

# TODO: Überprüfe die Daten auf Aktiensplits. 
# TODO: Überleg dir ein System um die Ergebnisse zu vergleichen bzw zu labeln. 

# TODO: holdings nennen wir handelsstrategien
# TODO; Ein-Periodenmodell um Umschichtungen zu vermeiden.

##################################################
# Dataloader
# DONE: Erstellung eines Dataloaders für die Daten
# DONE: Cashe auf single ticker basis speichern.
# DONE: TimebasedDataset erstellen.
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
# TODO: Im paper werden auch aktionen und cash als observations hinzugefügt.


####################################################
# Agenten
# DONE: DQN Agent
# DONE: MADDPG Agent
# DONE: Model Builder implementiert
# DONE: Loss function for MADDPG Agent
    # DONE: 2 It seems like all agents are learning the same thing. Maybe we somewhere loose the gradients? 
# TODO: MADDPG Agenten mit TIPP
# TODO: MADDPG Agenten mit CPPI
# TODO: UP Agent

# TODO: Methodik zum weitertrainieren der Agenten implementieren.

##################################################
# Risk Management
# TODO:
# Agent seems to trade too much -> causes a lot of trading costs.
# Impelment a filter? Or a epsilon region to trade? 
# Would it make sense to sample many actions at a state and add them to the replay buffer?


######################################################
# Logging and other Stuff
# DONE: Logging der Agenten
# DONE: Logging der Agenten in Tensorboard
# TODO: Report generator erstellen.


##################################################
# Generelle TODOs
# DONE: Tensorboard Logging
# DONE: Tensorboard Gewichtungen scheinen nicht zu funktionieren.
# TODO: CUDA, MPS, GPU Support
# DONE: Alles von listen und arrays auf tensors umstellen.
# TODO: Visualisierung Stock Preis und Anteil in einem Plot damit man 
# sieht wann der Agent genau kauft bzw verkauft.
# TODO: Hyperparameter Tuning
# TODO: Visualisierungen erstellen.
# TODO: Tracken der besten implementierungen. 
# TODO: Trainieren bis der total reward nicht mehr steigt.

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


###########################################################
# Critics about the Paper
# Actual Stocks are not mentioned. 
# Only short train and test time. 

