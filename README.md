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



[

sgn(number): provides the signs at the sink and source
]


[
permutation_sign(permuted_elements): The quarks in a correlator are presented at some point as Permutation_0 = [[q0, 0],[q1, 1],[q2, 2],...]. At that point, the numbers of quarks are extracted and given as a listm i.e. Permutation_0 -> [0, 1, 2..]. This function calculates the overall sign of a permutation based on the order of quark numbers in the list.
]



[
is_valid_contraction(candidate, Start_List):
 
[x for inner_list in candidate for x in inner_list] creates a new list that consists of all the elements from the inner lists contained within the candidate list.
Here, it is used to separate contracted quark pairs into individual quarks.
set(map(tuple, all_elements)) converts each element in all_elements into a tuple, and then all the tuples are added to a set.
A Wick contraction is considered valid if each unique quark in the correlator appears exactly once.
The function compares the length of the set consisting of initial quarks before contraction with the length of the set consisting of quarks after contraction.
If the lengths are the same, the contraction is valid and the function gives True.
]


[

Generate_Contraction(Start_List):
pairs=[list(pair) for pair in combinations(Start_List, 2)] generates a list where each element is a combination of two elements from Start_List. It includes all possible combinations while ignoring the order.
paar_contraction has the form [[quarki, i], [quarkj, j]].
In order for a contraction to be valid, quarki must be quark_j in bar (or other way). I.e. in our notation |quarki| = |quarkj| + 0.2 or  |quarki| + 0.2 = |quarkj|
So we exclude all pairs, which do not satisfy that condition! All possible pairs are saved in "possible_pairs"
We always must have an even number of quarks in the correlators, let's say 2N. 
 We now create the list pairs2 from possible_pairs. Each element of it is a combination of N elements from possible_pairs, meaning that each element in pairs2 could be a valid Wick contraction candidate.
The function checks the validity of each element in pairs2 using the previous function and removes the invalid ones.
]





[

Correlator(Sink, Source):
The function Correlator(Sink, Source) takes two lists of quarks, one at the sink and one at the source
the list at the sink and source are of the form:  [hadron_1, hadron_2,...]. where hadron_i can be e.g. Nucleonp or NucleonpB.
Correlator(Sink, Source) plots as an output the quarks at the Sink and Source with numbering at it gives all possible Wick contractions with the correpsonding numerical factor for each diagram.

]


[

Correlator_J(senken0, quelle0):

"hadron_multiplication "takes a list of hadrons and perform a cartesian product on them, this is needed in case we have at the sink and/or source Pion0 or Eta0.
 If we have "lenL" lists of quarks at the sink and "lenR" at the source, then we have in total "lenL * lenR" correlators
We calculate all of these correlators and put them all in the list "Total_Wick_Contraction0"
]



In the functions Correlator_J_S and Correlator_S we do the same as for "Correlator_J" and "Correlator" but here with simplifications using equation (4.78) in spectroscopy.pdf.
