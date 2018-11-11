# Summarizing-Scientific-Articles
## Data Format
```xml
<PAPER>
<TITLE> Similarity-Based Estimation of Word Cooccurrence Probabilities </TITLE>
<AUTHORLIST>
  <AUTHOR>Ido Dagan</AUTHOR>
  <AUTHOR>Fernando Pereira</AUTHOR>
  <AUTHOR>Lillian Lee</AUTHOR>
</AUTHORLIST>
<ABSTRACT>
  <A-S ID='A-0' AZ='BKG'> In many applications of natural language processing it is necessary to determine the likelihood of a given word combination . </A-S>
</ABSTRACT>
<BODY>
  <DIV DEPTH='1'>
    <HEADER ID='H-0'> Introduction </HEADER>
    <P>
      <S ID='S-0' AZ='BKG'> Data sparseness is an inherent problem in statistical methods for natural language processing . </S>
    </P>
  </DIV>
</BODY>
<REFERENCELIST>
<REFERENCE>
<SURNAME>Sugawara</SURNAME>, K., M. <SURNAME>Nishimura</SURNAME>, K. <SURNAME>Toshioka</SURNAME>, M. <SURNAME>Okochi</SURNAME>, and T. <SURNAME>Kaneko</SURNAME>.
<DATE>1985</DATE>.
Isolated word recognition using hidden Markov models.
In Proceedings of ICASSP, pages 1-4, Tampa, Florida. IEEE.
</REFERENCE>
</REFERENCELIST>
</PAPER>
```
  - annotated data is present in xml format.
  - iterate through each tag and extract relevant feature information.
