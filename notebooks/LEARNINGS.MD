# Learnings

## EDA

### Data
- Columns in train dataset: `id`, `prompt`, options `A` to `E`, and `answer`.
- Average prompt length is about 118 characters.

### Prompt
- Turns out the prompt has prefix, actual question and the suffix.

There are only 6 prefixes and 4 suffixes:
- Prefixes: Pick the best possible answer / Select the most accurate option / Identify the correct statement / Determine the correct option / Choose the correct answer / Which of the following is correct? (~200 prompts have no prefix)
- Suffixes: among the listed options / based on the given context / from the following choices / carefully (~538 prompts have no suffix)

### Duplicates
- Only 242 full prompts are exact duplicates.
- But once I strip the prefix and suffix, 1748 of the questions are duplicates. So most questions repeat, they were just dressed up with different prefixes/suffixes.

### Same question, but the answer text changes
- Each question always maps to the same answer letter across duplicates.
- But the text of the correct answer varies. The same question can have up to 9 different versions of the correct answer text.
- Example: "What is the Erlangen program?" has 9 different correct answer texts.
- The wording changes by small margins, similar words used.

### To do tom
- Embeddings / similarity should help here, both for matching duplicate questions and for matching the answer texts that only differ slightly.
- Inspect more on duplicates part.

Still remaining.

## TF-IDF + Logistic Regression

### Why did LR worked giving reasonably good score even if it doesn't have understanding?
- TF-IDF + LR has no concept of meaning.
- Rather it exploits the fact that the correct option tends to share more words/bigrams with the question than the wrong ones do.
- And it is proven because of roughly ~2x increase in the score from ~0.35 from random/base submission to ~0.75 with this.
- Helped clear the cutoff xD.

### Why did class weight worked inversely, so slightly?
- I overlooked a mistake and assigned class weights of {0:3, 1:1}, when infact label 0 makes 80% of the data, while label 1 makes 20%(Unfolded the options to individuals for Logistic regression to be trained upon).
- I inverted it but before I did so, I submitted it on kaggle and it raised a question.
- The faulty class weight gave a score of 0.7518 while the correct(better {0:1, 1:3}) gave slighlty less 0.7468, why?
- Turns out the ranking is mostly invariant to class weights and it is must certainly be noise.

## BERT + LR

### Why did BERT+LR score less than TFIDF+LR?
- BERT embeddings have semantic meaning/understanding unlike tfidfs.
- It scored lower because it might have struggled as in the options words used are semantically similar.

## BiLSTM from scratch
- built entire BiLSTM from scratch with bit of help
- the kaggle score vastly outperformed val score, meaning our val score is unreliable
- fix will be to implement k-folds cross validation
- current score disparity MAP@3 is `Val: 0.59` and `Kaggle: 0.71`