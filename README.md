# diaTM Dialectical Topic Models

diaTM is an implementation of the work of [Stephen Crain et al (2010).](https://www.ncbi.nlm.nih.gov/pubmed/21346955)
in which the authors describe a dialect sensitive topic modelling approach for
content analysis of content in more than one dialect (such as medical papers
and community fora).


## Installation

Run `pip install -r requirements.txt` and then `python setup.py install`.

## Usage

You can currently fit the model using something like this:

```python
from diatm import DiaTM

dtm = DiaTM(n_topic=10, n_dialects=2)

collection1 = [
  ['hello','blah','blah',...],
  ...
]


collection2 = [
  ['greetings','posho','posho',...],
  ...
]

X = [collection1, collection2]

dtm.fit(X)

```

It's pretty slow at the moment, I might be able to optimise the code one day.
