##################################################
# Wichtige TODOs
# DONE: Asset Tracker für beliebige anzahl an Episoden implementieren.
# DONE: Implementiere einen generallen Tracker.
# DONE: Überprüfe den Dataloader.
# DONE: optimize the code for MPS

# DONE: holdings nennen wir handelsstrategien bzw portfolio
# TODO; Ein-Periodenmodell um Umschichtungen zu vermeiden.

##################################################
# Dataloader
# DONE: Erstellung eines Dataloaders für die Daten
# DONE: Cashe auf single ticker basis speichern.
# DONE: TimebasedDataset erstellen.
# DONE: Alle ticker herausfinden.
# DONE: Datamanger implementieren. Sollte alle daten laden können. Vorhandene Daten ergänzen können. etc...
# TODO: Survival bias umgehen.

# DONE: Überprüfe die Daten auf Aktiensplits. 
# - Aktiensplits und Dividenden werden automatisch im adjusted Close Preis berücksichtigt. -> wir verwenden yfiance auto adjusted Close Preis.

##################################################
# Preprocessor
# DONE: Preprocessing der Daten
# DONE: Berechnung der technischen Indikatoren

###################################################
# Environment
# DONE: Single Agent Portfolio Environment
# DONE: Multi Agent Portfolio Environment mit Shared Observation
# DONE: Multi Agent Portfolio Environment mit Shared Observation und Shared Action
# DONE: Option implementieren für Agenten welche nicht nur den aktuellen State 
# sondern auch die Historie betrachten
# DONE: Im paper werden auch aktionen und cash als observations hinzugefügt.

# TODO: Balance wird nicht mehr berechnet, daher verbleibt ein eintrag mit 0 .


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
# TODO: Laden der Agenten implementieren.


######################################################
# Logging and other Stuff
# DONE: Logging der Agenten
# DONE: Logging der Agenten in Tensorboard
# TODO: Report generator erstellen.


##################################################
# Generelle TODOs
# DONE: Tensorboard Logging
# DONE: Tensorboard Gewichtungen scheinen nicht zu funktionieren.
# DONE: CUDA, MPS, GPU Support
# DONE: Alles von listen und arrays auf tensors umstellen.
# DONE: Visualisierung Stock Preis und Anteil in einem Plot damit man 
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
# DONE: Test für den DQN Agent
# DONE: Test für den MADDPG Agent

######################################
# Künftige Erweiterungen

# Andere Agenten:
# TODO: A2C Agent
# TODO: DDPG Agent mit TIPP
# TODO: Ensemble Agenten
# TODO: Ensemble Agenten mit TIPP
# TODO: Ensemble Agenten mit CPPI

# Visualisierung der final train balance


###########################################################
# Critics about the Paper
# Actual Stocks are not mentioned. 
# Only short train and test time. 



# Weitere Ideen:
# TODO Sortiono ratio als reward signal
