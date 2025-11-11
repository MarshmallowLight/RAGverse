# rag_qa_multitq.py
# QA template for MultiTQ — return EXACTLY one entity string (no extra words).
# Emphasis: subject/object orientation must not be swapped; apply temporal operators precisely.

rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze multiple triple facts and corresponding questions with time constraints meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    "Keep subject/object orientation. Match the same base relation. Apply temporal operator precisely."
    'Conclude with "Answer: " to present return 10 short answer candidates ranked best-to-worst, devoid of additional elaborations.'    
)

# Minimal one-shot example to cement format and S/O orientation + temporal operator 'before'
one_shot_passages = (
    "Event A: On 2010-08-30, European Central Bank criticized Romania \n"
    "Event B: On 2011-02-14, European Central Bank criticized government of Germany\n"
)

one_shot_question = (
    "Question: Before Germany, who did the European Central Bank criticize last?\n"
    "Answer:"
)

one_shot_answer = "Romania"

# Final template: system rule -> (optional) one-shot -> runtime user prompt
# ${prompt_user} is filled by the pipeline with retrieved passages + the actual question, and must end with 'Answer:'.
prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_passages + one_shot_question},
    {"role": "assistant", "content": one_shot_answer},
    {"role": "user", "content": "${prompt_user}"}
]

