##################################################################
# Questions to be answered
##################################################################

# TODO: Sollten wir indikatoren als features verwenden?
# - Wenn ja welche?


# TODO: Wie sind die Renditen der Wertpapiere genau definiert? 




# TODO: Vergleich vorbereiten für mit und ohne Kosten. 















####################################################################
# Questions that are already answered
####################################################################

# TODO: Which stock assortment to use? 
# - Desmettre: Ein möglichst Balanciertes Assortment aus Aktien durch verschiedene Sektoren und Ländern. Etwa 100-500 Stk.

# TODO: Welche Reward funktion? log reward? oder nur stepreward? 
# - Desmettre: absolute Rendite/absolute return sollte fürs erste passen. 
# - Später könnte man auf log return umstellen. (In diesem Fall meint der Return aber die Rendite



####################################################################
# Generelle Notizen
####################################################################

# TODO: Holdings nennen wir eigentlich Handelsstrategien
# TODO: Actions sind in der Finanzmathematik eigentlich als portfoliovector bekannt. 



####################################################################
# TODO: Hypothesen: 
# Die Größe des Buffers verhält sich wie eine Informationskapazität welche die Informationen der Agenten speichert. 
# Um zu lernen samplen wir aus dem Buffer. 
# Das bedeutet, dass man den Buffer als parallel kapazität sehen kann. 
# Eine kleinere Kapzität bedeutet das der Agent über die Trainingsinformationen instabiler lernt.
# Also das der finale Portfoliowert mehr Schwanken wird. Der Grund dafür ist das gute Informationen sparse sind. 
# Man sollte also einen Buffer nehmen der darauf achtet das der eine bestimmte Rewardverteilung aufrecht erhält.


