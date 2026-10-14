# Task 1 - Prompting frameworks

I selected RACE, CLEAR and CRISP from the
[Frompting framework library](https://frompting.com/frameworks). I first
generated drafts there, then corrected them for the restaurant SQL task.

## RACE

[RACE](https://frompting.com/frameworks/race) separates Role, Action,
Context and Expectation. I chose it because the database context and
required SQL output can be stated clearly. I expect it to reduce ambiguity.

## CLEAR

[CLEAR](https://frompting.com/frameworks) covers Clarity, Language,
Examples, Audience and Requirements. I chose it to make the SQL syntax,
example and evaluator requirements explicit. I expect more consistent
queries and fewer formatting errors.

## CRISP

[CRISP](https://frompting.com/frameworks/crisp) separates Context, Role,
Intent, Style and Parameters. I chose it because it supports both the
translation task and its constraints. I expect controlled SQL output.

The three files in `prompts/` contain the same task information, schema,
relationships, output rules and one original demonstration. Only their
section organisation differs. The demonstration makes all three one-shot,
rather than zero-shot. `{{question}}` is replaced for each input question.
The required output columns are house number and restaurant name.
Reference rows are never supplied to the model. These expectations are
hypotheses; Task 2 measures the actual performance.
