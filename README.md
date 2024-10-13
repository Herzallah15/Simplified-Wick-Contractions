# Simplified-Wick-Contractions

Here are some informations about the definitions used in our code:

up-quark =  1, \bar up-quark = 1.2

down-quark =  2, \bar down-quark = 2.2

strange-quark =  3, \bar strange-quark = 3.2

To define a baryon operator we use the definitions given in equation (4.77) in spectroscopy.pdf. In the definition we do not include the d coefficients. However, during the simplification, the code makes use of the relations between them. Hence, a proton is defined as: nucleonp = [1, [1, 1, 2]], where the first element in the list corresponds to the overall factor. If we want to use the code with simplification, we have to use another definition for the proton, namely as: Nucleonp = ["n", nucleonp], where the role of "n" is to tell the code it should use for the simplification the equation: $d_{\alpha \beta \gamma}^N = d_{\beta \alpha \gamma}^N$.

To define meson operators, we use the equations given in Chapter 5 in spectroscopy.pdf. For example equation (5.26) for the eta0. An operator with a superposition is defined as [[coefficient_term_1, [flavor_structure_1]], [coefficient_term_2, [flavor_structure_2]], ...]. For example the eta0 is defined as [[1, [2.2, 2]], [1, [1.2, 1]]].



#[sgn(number): provides the signs at the sink and source]


#[permutation_sign(permuted_elements): The quarks in a correlator are presented at some point as Permutation_0 = [[q0, 0],[q1, 1],[q2, 2],...]. At that point, the numbers of quarks are extracted and given as a listm i.e. Permutation_0 -> [0, 1, 2..]. This function calculates the overall sign of a permutation based on the order of quark numbers in the list.]

#[is_valid_contraction(candidate, Start_List): [x for inner_list in candidate for x in inner_list] creates a new list that consists of all the elements from the inner lists contained within the candidate list. Here, it is used to separate contracted quark pairs into individual quarks. set(map(tuple, all_elements)) converts each element in all_elements into a tuple, and then all the tuples are added to a set. A Wick contraction is considered valid if each unique quark in the correlator appears exactly once. The function compares the length of the set consisting of initial quarks before contraction with the length of the set consisting of quarks after contraction. If the lengths are the same, the contraction is valid and the function gives True.]


#[Generate_Contraction(Start_List): pairs=[list(pair) for pair in combinations(Start_List, 2)] generates a list where each element is a combination of two elements from Start_List. It includes all possible combinations while ignoring the order. paar_contraction has the form [[quarki, i], [quarkj, j]]. In order for a contraction to be valid, quarki must be quark_j in bar (or other way). I.e. in our notation |quarki| = |quarkj| + 0.2 or  |quarki| + 0.2 = |quarkj| So we exclude all pairs, which do not satisfy that condition! All possible pairs are saved in "possible_pairs" We always must have an even number of quarks in the correlators, let's say 2N. We now create the list pairs2 from possible_pairs. Each element of it is a combination of N elements from possible_pairs, meaning that each element in pairs2 could be a valid Wick contraction candidate. The function checks the validity of each element in pairs2 using the previous function and removes the invalid ones.]





#[ Correlator(Sink, Source): The function Correlator(Sink, Source) takes two lists of quarks, one at the sink and one at the source the list at the sink and source are of the form:  [hadron_1, hadron_2,...]. where hadron_i can be e.g. Nucleonp or NucleonpB.Correlator(Sink, Source) plots as an output the quarks at the Sink and Source with numbering at it gives all possible Wick contractions with the correpsonding numerical factor for each diagram.]


#[Correlator_J(senken0, quelle0):"hadron_multiplication "takes a list of hadrons and perform a cartesian product on them, this is needed in case we have at the sink and/or source Pion0 or Eta0. If we have "lenL" lists of quarks at the sink and "lenR" at the source, then we have in total "lenL * lenR" correlators We calculate all of these correlators and put them all in the list "Total_Wick_Contraction0"]



#In the functions Correlator_J_S and Correlator_S we do the same as for "Correlator_J" and "Correlator" but here with simplifications using equation (4.78) in spectroscopy.pdf.

<font size="15">CORRELATORS IN TERMS OF SINGLE-HADRON OPERATORS</font>


In general, to obtain the unsimplified Wick contractions, use small letters for the names of the hadrons. The Wick contractions are then obtained from the function "Correlator_J". To obtain the simplified Wick contractions, use a capital letter at the beginning of the names of the hadrons. The Wick contractions are then obtained from the function "Correlator_J_S". 


Examples:

####################################################################################


To obtain the contractions for $N^+ \Pi^-$ at the sink and $\bar N^0$ at the source, i.e. $< N^+ \Pi^- \bar N^0>$, we define:



```python
sink = [nucleonp, pim] 
source = [nucleon0B] 
result = Correlator_J(sink, source)
for i, x in enumerate(result):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])


# For the simplified version we define:


sink = [Nucleonp, Pim] 
source = [Nucleon0B] 
result = Correlator_J_S(sink, source)
for i, x in enumerate(result):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])


```
####################################################################################


To obtain the contractions for $\Delta^{++} \Pi^-$ at the sink and $\bar\Delta^+$ at the source, i.e. $<\Delta^{++} \Pi^- \bar\Delta^+>$, we define:


```python


senken0 = [deltapp, pim]
quelle0 = [deltapB]
resultx = Correlator_J(senken0, quelle0)
for i, x in enumerate(resultx):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])



# For the simplified version we define:

senken0 = [Deltapp, Pim]
quelle0 = [DeltapB]
resultx = Correlator_J_S(senken0, quelle0)
for i, x in enumerate(resultx):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])


```


####################################################################################


To obtain the contractions for $\Delta^{++}~ N^0 ~ \Pi^-$ at the sink and $\bar\Delta^{+} ~ \bar N^0 ~ \bar\Pi^0$ at the source, i.e. $<\Delta^{++} ~N^0 ~\Pi^- ~\bar\Delta^{+} ~\bar N^0 ~\bar\Pi^0>$, we define:


```python
#It takes from 1 to 2 minutes to give the contractions
senken0 = [deltapp, nucleon0, pim]
quelle0 = [deltapB, nucleon0B, pi0]
resultx = Correlator_J(senken0, quelle0)
for i, x in enumerate(resultx):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])


# For the simplified version we define:

#It takes from 1 to 2 minutes to give the contractions
senken0 = [Deltapp, Nucleon0, Pim]
quelle0 = [DeltapB, Nucleon0B, Pi0]
resultx = Correlator_J_S(senken0, quelle0)
for i, x in enumerate(resultx):
    print(f"Diagram Number:{i + 1} ", x[0], x[1])

```
