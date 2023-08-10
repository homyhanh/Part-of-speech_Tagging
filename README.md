# Part-of-speech_Tagging
Part-of-speech tagging (POS tagging) is the task of tagging a word in a text with its part of speech. A part of speech is a category of words with similar grammatical properties. 

This notebook contains code for Transformation Based Learning that can tag POS in an English sentence.

Example

| John | lives | in | London | . |
|------|-------|------|-------|-------|
| NNP | VBZ | IN | NNP | . | 

# Datasets
There are 45 sentences that are manually labeled based on the Penn Treebank tagset. 

<img src="https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/82110bba-204a-4f85-8e23-07732dc5fe4e" alt="..." width="800" />

# Transformation-Based Learning
## Training time: learn transformations
1. Initialize each instance in the training data with an initial annotator
2. Consider all the possible transformations, and choose the one with the highest score. 
3. Append it to the transformation list and apply it to the training corpus to obtain a “new” corpus.
4. Repeat steps 2-3.
## Testing time: applying transformations
1. Initialize each example in the test data with the same initial annotator
2. Apply the transformations in the same order as they were learned.
## TBL for POS tagging 
• The initial state-annotator: most common tag for a word according to the training data

• The space of allowable transformations
– Rewrite rules: change cur_tag from X to Y.

– Triggering environments (feature types): unlexicalized or lexicalized

 
