import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt
import numpy as np

Triple = Tuple[str, str, str]
Quad   = Tuple[str, str, str, Optional[str]]

class Fact(BaseModel):
    # 允许 [s,r,o] 或 [s,r,o,t]，其中 t 可以是 None / null / ""：
    fact: List[Union[Triple, Quad]] = Field(
        description="Facts as [s,r,o] or [s,r,o,t]; t may be null/None."
    )
# A minimal, self-contained temporal-aware LLM filter leveraging your existing llm_model
class TemporalAwareFilter:
    def __init__(self, llm_model, message_template=None):
        self.llm_model = llm_model
        # re-use the same system prompt style you use elsewhere; keep it short and deterministic
        self.message_template = message_template or [
            {"role": "system", "content":
             "You are a precise filter. You will ONLY select from the provided fact list. "
             "Do NOT paraphrase any fact. Copy chosen facts EXACTLY as shown."}
        ]
        self.input_template = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## constraints ## ]]\n"
            "subject_hint: {subject_hint}\n"
            "relation_aliases: {relation_aliases}\n"
            "temporal_constraint: {temporal_constraint}\n"
            "anchor_object_hint: {anchor_hint}\n"
            "selection_rules:\n"
            "- Only pick facts whose subject == subject_hint and relation in relation_aliases.\n"
            "- If an ANCHOR fact (subject_hint, any alias, anchor_object_hint, time) exists, call it ANCHOR.\n"
            "- For 'before': select only facts with time < ANCHOR.time and prefer the latest ones (closest to ANCHOR).\n"
            "- For 'after': select only facts with time > ANCHOR.time and prefer the earliest ones (closest to ANCHOR).\n"
            "- Facts without time should be deprioritized and used only if needed to reach K.\n"
            "- Output EXACTLY {k} facts and copy them EXACTLY as in the list below.\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "[[ ## output ## ]]\n"
            "{{\"fact_after_filter\": {json_hint}}}"
        )

    def _call(self, question, fact_before_filter, k, subject_hint, relation_aliases, temporal_constraint, anchor_hint):
        messages = deepcopy(self.message_template)
        json_hint = "[[\"s\",\"r\",\"o\",\"t\"], ...]"  # guide the JSON shape
        messages.append({
            "role": "user",
            "content": self.input_template.format(
                question=question,
                subject_hint=subject_hint,
                relation_aliases=relation_aliases,
                temporal_constraint=temporal_constraint,
                anchor_hint=anchor_hint,
                k=k,
                fact_before_filter=fact_before_filter,
                json_hint=json_hint
            )
        })
        # Your llm_model.infer returns (response_message, metadata, cache_hit)
        resp, _meta, _cache = self.llm_model.infer(messages)
        return resp

    @staticmethod
    def _parse_json_block(text):
        # Try to extract a JSON object containing "fact_after_filter"
        try:
            obj = json.loads(text)
            if "fact_after_filter" in obj:
                return obj["fact_after_filter"]
        except Exception:
            pass
        # Loose fallback: find the last json-like block
        m = re.findall(r'\{.*\}', text, flags=re.S)
        for chunk in reversed(m):
            try:
                obj = json.loads(chunk)
                if "fact_after_filter" in obj:
                    return obj["fact_after_filter"]
            except Exception:
                continue
        return []

    def rerank(self, question, candidate_items_4, k, subject_hint, relation_aliases, temporal_constraint, anchor_hint):
        """
        candidate_items_4: List[(s,r,o,t)], already normalized and deduped as you prefer.
        Returns: List[(s,r,o,t)] chosen by the LLM (may be < k; caller should backfill).
        """
        # Convert candidates into a deterministic plain list for the model
        # Format each fact uniformly: "(s) -r-> (o) @time=YYYY-MM-DD" or "@time=None"
        lines = []
        for (s,r,o,t) in candidate_items_4:
            lines.append(f"({s}) -{r}-> ({o}) @time={t}")
        fact_str = "\n".join(lines)

        resp = self._call(
            question=question,
            fact_before_filter=fact_str,
            k=k,
            subject_hint=subject_hint or "",
            relation_aliases=json.dumps(sorted(list(relation_aliases or []))),
            temporal_constraint=temporal_constraint or "",
            anchor_hint=anchor_hint or ""
        )

        picked = self._parse_json_block(resp)
        out = []
        for it in picked:
            try:
                if isinstance(it, (list, tuple)):
                    # accept 3 or 4, normalize to 4
                    if len(it) == 4:
                        s,r,o,t = it
                    elif len(it) == 3:
                        s,r,o = it; t = None
                    else:
                        continue
                else:
                    tup = ast.literal_eval(it)
                    if len(tup) == 4:
                        s,r,o,t = tup
                    elif len(tup) == 3:
                        s,r,o = tup; t = None
                    else:
                        continue
                out.append((str(s), str(r), str(o), None if t in (None,"None","null") else str(t)))
            except Exception:
                continue
        # dedupe while preserving order
        seen, uniq = set(), []
        for f in out:
            if f not in seen:
                uniq.append(f); seen.add(f)
        return uniq[:k]
class DSPyFilter:
    def __init__(self, hipporag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self, response):
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # call openai

        self.default_gen_kwargs['max_completion_tokens'] = 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print('exception', e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print('result_indices exception', e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}