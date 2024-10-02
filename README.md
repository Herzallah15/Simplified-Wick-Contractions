# Simplified-Wick-Contractions

#To obtain the unsimplified Wick contractions, use small letters for the names of the hadrons. The Wick contractions are then obtained from the function "Correlator_J"

#For example, to obtain the contractions for $N^+ \Pi^-$ at the sink and $\bar N^0$ at the source, i.e. $< N^+ \Pi^- \bar N^0>$, we define:

sink = [nucleonp, pim] 

source = [nucleon0B] 

result = Correlator_J(sink, source)

for i, x in enumerate(result):

 print(f"Diagram Number:{i + 1} ", x[0], x[1])

#To obtain the simplified Wick contractions, use a capital letter at the beginning of the names of the hadrons. The Wick contractions are then obtained from the function "Correlator_J_S". The simplified Wick contractions from the above example are abtained by:

sink = [Nucleonp, Pim] 

source = [Nucleon0B] 

result = Correlator_J(sink, source)

for i, x in enumerate(result):

 print(f"Diagram Number:{i + 1} ", x[0], x[1])

