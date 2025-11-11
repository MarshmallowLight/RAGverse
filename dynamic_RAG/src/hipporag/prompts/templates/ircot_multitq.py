# ircot_multitq.py

ircot_system = (
    "You are an assistant for interleaving retrieval with reasoning (IRCoT).\n"
    "Protocol:\n"
    "- At each step output EXACTLY ONE line.\n"
    "- Use 'Search: <concise query>' (<=10 tokens) to request more passages.\n"
    "- When sufficient, output 'Answer: <single string>' and stop.\n"
    "Constraints:\n"
    "1) SUBJECT/OBJECT ORIENTATION: Do not swap S and O. If the question asks "
    "'<SUBJECT> did <REL> to/with which country?', keep <SUBJECT> as S and return the OBJECT country.\n"
    "2) TEMPORAL OPERATORS: last=latest date; first=earliest; "
    "   before X: choose (S,R,O) with t<t_X closest to t_X; after X: t>t_X closest to t_X.\n"
    "3) DATE QUESTIONS: If the question asks 'When/What date', answer as YYYY-MM-DD using passage dates.\n"
    "4) OUTPUT FORMAT: No explanations, no punctuation beyond the required line.\n"
    "5) If evidence is insufficient, answer the most possible answer."
)


# Minimal one-shot demo (toy; keep it short)
one_shot_docs = (
    "Fact: Example1\nOman criticized CountryX on 2010-08-30 according to the report.\n\n"
    "Fact: Example2\nCountryX is a sovereign country.\n\n"
)

one_shot_user = (
    f"{one_shot_docs}"
    "Question: Before South Korea, with whom did Oman last wish to establish diplomatic cooperation? \nThought:"
)

one_shot_assistant = (
    "Answer: Qatar"
)

# Final prompt: system + (optional) one-shot + runtime user turn
prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": one_shot_user},
    {"role": "assistant", "content": one_shot_assistant},
    {"role": "user", "content": "${prompt_user}"}
]
