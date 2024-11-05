# ⭐ SINGLE-HADRON OPERATORS ⭐


All single-hadron operators are taken from spectroscopy.pdf (page 163 for Baryons):

**Δ⁺⁺** = Deltapp,
**Δ̅⁺⁺** = DeltappB,
**Δ⁺** = Deltap,
**Δ̅⁺** = DeltapB,
**Δ⁰** = Delta0,
**Δ̅⁰** = Delta0B,
**Δ⁻** = Deltam,
**Δ̅⁻** = DeltamB.

**N⁺** = Nucleonp,
**N̅⁺** = NucleonpB,
**N⁰** = Nucleon0,
**N̅⁰** = Nucleon0B.


**Σ⁺** = Sigmap,
**Σ̅⁺** = SigmapB,
**Σ⁰** = Sigma0,
**Σ̅⁰** = Sigma0B,
**Σ⁻** = Sigmam,
**Σ̅⁻** = SigmamB.


**Ξ⁰** = Xi0,
**Ξ̅⁰** = Xi0B,
**Ξ⁻** = Xim0,
**Ξ̅⁻** = XimB.


**Λ⁰** = Lambda0,
**Λ̅⁰** = Lambda0B.



**Ω⁻** = Omegam,
**Ω̅⁻** = OmegamB.



**π⁺** = Pip,
**π̅⁺** = PipB,
**π⁰** = Pi0,
**π̅⁰** = Pi0B,
**π⁻** = Pim,
**π̅⁻** = PimB.




**η⁰** = Eta0,
**η̅⁰** = Eta0B.



**K⁺** = Kaonp,
**K̅⁺** = KaonpB,
**K⁻** = Kaonm,
**K̅⁻** = KaonmB.





**Φ⁰** = Phi0,
**Φ̅⁰** = Phi0B.

# ⭐ TWO-HADRON OPERATORS ⭐



All two-hadron operators are also taken from spectroscopy.pdf (from section 7.2). To specify the two-hadron operator we always need to specify:

1- The representation: **I<sub>A</sub> × I<sub>B</sub>** = [**I<sub>A</sub>**, **I<sub>B</sub>**] = rep. The order of **I<sub>A</sub>**  and **I<sub>B</sub>** must be taken exactly as given in spectroscopy.pdf-section 7.2.




2- The total isospin **I** and the third component **I3**.





3-The particles **A**, **B** **∈** {'Delta', 'Nucleon', 'Sigma', 'Xi', 'Lambda', 'Omega', 'Pi', 'Eta', 'Kaon', 'KaonB','Phi'}.




4-The two-hadron operator (in \bar) is given now by: two_hadron_operatorB(rep, **I**, **I3**, **A**, **B**).





5-The two-hadron operator without \bar: two_hadron_operator(rep, **I**, **I3**, **A**, **B**).


# ⭐ THREE-HADRON OPERATORS ⭐




All three-hadron operators were derived by Fernando Alvarado. To specify the three-hadron operator we always need to specify:

1- The representation: **I<sub>A</sub> × I<sub>B</sub> × I<sub>C</sub>** = [**I<sub>A</sub>**, **I<sub>B</sub>**, **I<sub>C</sub>**] = rep. The order of **I<sub>A</sub>**, **I<sub>B</sub>** and **I<sub>C</sub>** must be taken exactly as specifed by Fernando Alvarado.




2- The total isospin **I**, the third component **I3** and the total isospin of **A** and **B** **I<sub>AB</sub>**.





3-The particles **A**, **B**, **C** **∈** {'Delta', 'Nucleon', 'Sigma', 'Xi', 'Lambda', 'Omega', 'Pi', 'Eta', 'Kaon', 'KaonB','Phi'}.




4-The three-hadron operator (in \bar) is given now by: three_hadron_operatorB(rep, **I**, **I3**, **I<sub>AB</sub>**, **A**, **B**, **C**).





5-The three-hadron operator without \bar: three_hadron_operator(rep, **I**, **I3**, **I<sub>AB</sub>**, **A**, **B**, **C**).
