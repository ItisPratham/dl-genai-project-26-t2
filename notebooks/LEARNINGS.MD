# Learnings

## 23rd June 2026

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