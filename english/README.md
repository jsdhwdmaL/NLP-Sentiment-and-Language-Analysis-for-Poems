## Introduction
- This product (1) extracts and classifies imageries and (2) classifies sentiments for Renaissance poems written in English.

## Procedure
1. Train raw data using feature engineering
2. Use Stanford CoreNLP to extract dependency parses
3. From the dependency parses, extract imageries and corresponding descriptive words -- *extraction of imagery done*
4. Train a basic sentiment model using English sentences data base
5. Use the trained model to classify imageries' sentiment into "positive" and "negative" classes
6. Calculate the number of words in each class
7. Add the number to feature engineering
8. Compare the model accuracy before and after the information of imagery sentiment

## Guide
- Implement "processing.ipynb" on raw data to do data analysis
- Extract dependency parses

## Tools Used
- NLTK
    - Bird, Steven, Edward Loper and Ewan Klein. (2009). Natural Language Processing with Python. O’Reilly Media Inc.
- Stanford CoreNLP
## Sources:
### Raw Dataset:
- https://www.kaggle.com/ultrajack/modern-renaissance-poetry
### Preprocessing:
- https://builtin.com/machine-learning/nlp-machine-learning
- https://studymachinelearning.com/text-preprocessing-removal-of-punctuations/
### Basic Imagery Extraction:
- https://kavita-ganesan.com/python-keyword-extraction/#.YmabwPNBzdp
- https://machinelearningknowledge.ai/tutorial-on-pos-tagging-and-chunking-in-nltk-python/
### Coreference Resolution and CoreNLP
- http://nlpprogress.com/english/coreference_resolution.html
- https://github.com/josubg/CorefGraph
- https://nafigator.readthedocs.io/en/latest/index.html
- https://stanfordnlp.github.io/CoreNLP/index.html
- Cite:
    - Rodrigo Agerri, Josu Bermudez and German Rigau (2014): "IXA pipeline: Efficient and Ready to Use Multilingual NLP tools", in: Proceedings of the 9th Language Resources and Evaluation Conference (LREC2014), 26-31 May, 2014, Reykjavik, Iceland.
    - Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.
### Feature Engineering
- https://www.analyticsvidhya.com/blog/2021/04/a-guide-to-feature-engineering-in-nlp/#:~:text=Feature%20engineering%20in%20NLP%20is%20understanding%20the%20context,classification%20task%20with%20and%20without%20doing%20feature%20engineering
- https://github.com/mohdahmad242/Feature-Engineering-in-NLP
- https://mlnlp.readthedocs.io/en/latest/Feature-engineering.html
- https://www.analyticsvidhya.com/blog/2020/11/understanding-naive-bayes-svm-and-its-implementation-on-spam-sms/#:~:text=1.%20Multinomial%20Naïve%20Bayes.%202.%20SVM.%20By%20seeing,predict%20if%20the%20message%20is%20spam%20or%20not.
### Training Basic Sentiment Model
- https://towardsdatascience.com/nlp-sentiment-analysis-for-beginners-e7897f976897
- https://blog.csdn.net/Itsme_MrJJ/article/details/123830831
### Other available sources:
- https://www.andyfitzgeraldconsulting.com/writing/keyword-extraction-nlp/
- https://github.com/andybywire/nlp-text-analysis/blob/master/text-analytics.ipynb
- https://medium.com/algoanalytics/automatic-labelling-of-text-for-nlp-5270e70a2f5f
