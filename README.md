# Simplified-Wick-Contractions

#To obtain the unsimplified Wick contractions, use small letters for the names of the hadrons. The Wick contractions are then obtained from the function "Correlator_J"
#For example, to obtain the contractions for N^+ \Pi^- at the sink and \bar Nucleon^0 at the source, i.e. < N^+ \Pi^- \bar Nucleon^0>, we define:
sink = [nucleonp, pim] 
source = [nucleon0B] 
result = Correlator_J(sink, source)
for i, x in enumerate(result):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])
