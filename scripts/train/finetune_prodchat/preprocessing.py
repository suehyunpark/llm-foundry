# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

r"""Example custom preprocessing function.

This is here to help illustrate the way to set up finetuning
on a local dataset. One step of that process is to create
a preprocessing function for your dataset, and that is what
is done below. Check out the LLM Finetuning section of
`../README.md` for more context.

For this example, we're going to pretend that our local dataset
is `./train.jsonl`.

Note: this dataset is actually a copy of one of our ARC-Easy
multiple-choice ICL eval datasets. And you would never actually
train on eval data! ... But this is just a demonstration.

Every example within the dataset has the format:
{
    'query': <query text>,
    'choices': [<choice 0 text>, <choice 1 text>, ...],
    'gold': <int> # index of correct choice
}

To enable finetuning, we want to turn this into a prompt/response
format. We'll structure prompts and responses like this:
{
    'prompt': <query text>\nOptions:\n - <choice 0 text>\n - <choice 1 text>\nAnswer: ,
    'response': <correct choice text>
}
"""

PROMPT_FORMAT = (
    'Respond to this question using the given context. \n'
    'Question:\n{title} {body}\n\n '
    'Context: \n{context}\n'
)

def prodchat_preprocessing_function(inp: dict):
    title = inp['title']
    body = inp['body']
    context = '\n'.join([ctx['text'][len(ctx['title'])+2:] for ctx in inp['ctxs']])  # remove title from text
    comment = inp['comments'].strip()  # text only
    # comment = inp['comments'][0]['body']
        
    return {
        'prompt': PROMPT_FORMAT.format(title=title, body=body, context=context),
        'response': comment
    }