Text Oriented Active Learning
=============================

This active learning toolbox can be used to 

1. Bootstrap supervised NLP tasks by intelligently selecting data for annotation, and
2. Conduct (simulated) active learning experiments using different sampling and machine learning heuristics.

**This is a very early stage software and, as such, it is subject to change.**

Setup
-----

TOAL requires Python 3.5 or higher, all the other dependencies are available in the requirements.txt files and can be installed using `pip3 install -r requirements.txt` (a virtual environment is not necessary but receommended).


Project Structure
-----------------

Folder organization is subject to change.

```
/
├── data                       -> Folder with test data
├── notebooks                  -> Dunno
├── test                       -> Test code
└── toal                       -> Main souce code
    ├── extractors                -> Package with all the available extractors
    ├── learners                  -> Package with all the available learners
    └── stores                    -> Package with all the available stores
```

