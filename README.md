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

# Handle unknown words
1. Random POS – tag: randomize the label among all existing labels
2. Most probable POS–tag: find the most frequent POS-tag and assume that unknown words always have this POS-tag. 
3. Overall POS distribution: the probability of an unknown word considering different labels will be similar to the probability distribution of the labels computed in the training set. In other words, the probability of an unknown word labeled q will be equal to the probability of a known word labeled q.
4. Hapax legomena: the distribution of POS-tags over unknown words is similar to the distribution over words that occur only once in the training set. These words are known as hapax legonema.
5. Regex tagger: use regular expressions to find the parts that need to be tagged.
# Result

Template

![image](https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/dfaee1e3-308b-4902-9a0a-646baa07ddab)

Rules are learned from TBL

![image](https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/d86d3c75-cf4e-465e-9c4a-30ce57830341)


Use accuracy to evaluate the model including all tags, known tags, and unknown tags.

- Without TBL algorithm

![image](https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/97e2c9fc-4d34-4439-9583-4c4f46eb1ae0)

- With TBL algorithm

![image](https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/3224adbe-549c-44b6-8ac7-ae0809554426)

## Example:

![image](https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/540c0556-644b-4270-b3e8-2a1bc16515b1)


