## Introduction
- ...
## Procedure
1. Train raw data using feature engineering
2. Use SuPar-Kanbun to extract dependency parses
3. From the dependency parses, extract imageries and corresponding descriptive words -- *extraction of imagery done*
4. Train a basic sentiment model using Chinese sentences data base
5. Use the trained model to classify imageries' sentiment into "positive" and "negative" classes
6. Calculate the number of words in each class
7. Add the number to feature engineering
8. Compare the model accuracy before and after the information of imagery sentiment
## Guide:
- ...
## Sources:
### Preprocessing:
- https://github.com/jiaeyan/Jiayan
### Parsing
- https://github.com/KoichiYasuoka/SuPar-Kanbun
### Other Available Sources:
- https://docs.cltk.org/en/latest/index.html
