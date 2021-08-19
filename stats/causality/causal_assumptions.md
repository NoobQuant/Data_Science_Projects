# List of different causal assumptions used in the literature

 - **Consistency**
   - Formulation (Pearl, 2009, p. 99): $X=x \implies Y(x, u) = Y$ for every $u$. That is, if the actual value of exposure X turned out to be x, then the value of that outcome Y would take (if X were x) is equal to the actual value of Y. 
   - "*Assumption that is violated whenever versions of treatment/exposure have unintended side effects on the outcomes of interest*" (Pearl 2010).
   - "Consistency is generally satisfied if we can describe a well-defined intervention that reflects the way in which the exposure changed in our data" (Caniglia & Murray 2020).

 - **Exchangeability**
   - Loose formulation: "*You can exchange the treated and the untreated for the same result*" ([Data Talks, Exchangability: Part 1](https://www.youtube.com/watch?v=iUZA5dTgegQ)).
   "*[We] ought to make sure that any response differences between the treated and the untreated group is due to the treatment itself and not to some intrinsic differences between the groups that are unrelated to the treatment*" (Pearl 2009, p.196).
   - For more formal treatment, see Saarela et al (2020).

 - **Ignorability**
   - TBF

 - **No carryover effect**
   - Used in panel data setting in Imai & Kim (2019).
   - Assumption that past treatments do not directly affect current outcome.
   - $Y_{it}(X_{i1}, X_{i2}, \dots , X_{it}) = Y_{it}(X_{it})$

## Sources

 - Caniglia & Murray (2020): Difference-in-difference in the time of cholera: a gentle introduction for epidemiologists
 - Imai & Kim (2019): [When should we use unit fixed effects regression models for causal inference with longitudinal data?](https://imai.fas.harvard.edu/research/files/FEmatch.pdf)
 - Pearl (2009): Causality
 - Pearl (2010): [On the Consistency Rule in Causal Inference: Axiom, Definition, Assumption, or Theorem?](https://ftp.cs.ucla.edu/pub/stat_ser/r358.pdf)
 - Saarela, Stephens and Moodie (2020): [The role of exchangeability in causal inference](https://arxiv.org/pdf/2006.01799.pdf)
