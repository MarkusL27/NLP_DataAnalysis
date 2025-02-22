# NLP_DataAnalysis  
  
### Conda Environment erstellen mit YAML File:  
  
```  
conda env create --f environment.yml  
```  

### Verwendung des Codes:

- main.py ausführen
- Alle Funktionnsaufrüfe sind in der main() Funktion enthalten
- Aus Gründen der Performance sind die beiden Funktionen zur Ermittlung der optimalen Themenanzahl für LDA / LSA auskommentiert. 

### Ablauf

- Zunächst werden die Daten aus der CSV Datei geladen und als Pandas Dataframe eingelesen, wobei nur die für das Projekt relevante Spalte 'Reports' betrachtet wird.
- Es wird ein Bag-Of-Words sowie ein TF-IDF Vektor erstellt
- Die beiden Vektoren werden miteinander verglichen, indem die vertikale Summe der jeweiligen Spalten gebildet werden und daraus die Top 10 Wörter ermittelt und für beide Vektoren ausgegeben
- (Auskommentiert): Methoden zur Ermittlung der optimalen Anzahl an Themen für LDA / LSA werden aufgerufen, optimale Anzahl wird ausgegeben
- Themenextraktion mittels LDA und LSA wird durchgeführt, Themen werden ausgegeben

### weitere Dokumentation

im Code wurden die Funktionen mithilfe von Docstrings genauer dokumentiert