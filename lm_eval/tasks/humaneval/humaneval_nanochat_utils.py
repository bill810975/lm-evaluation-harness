import re


def extract_imports(prompt: str) -> str:
    """Grab leading import/from lines from the original prompt."""
    imports = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def extract_program(completion: str) -> str:
    """
    Return the first code block if fenced, otherwise the raw completion.
    Mirrors nanochat.tasks.humaneval.extract_program.
    """
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If there's an unmatched fence, take everything before it.
    if "```" in completion:
        return completion.split("```", 1)[0].strip()
    return completion.strip()


def build_predictions_nanochat(resps, docs):
    """
    Build runnable candidates similar to nanochat eval:
    prepend leading imports from prompt, keep only the code portion of the completion.
    """
    out = []
    for resp, doc in zip(resps, docs):
        imports = extract_imports(doc["prompt"])
        cand = []
        for r in resp:
            code = extract_program(r)
            program = f"{imports}\n\n{code}" if imports else code
            cand.append(program)
        out.append(cand)
    return out
