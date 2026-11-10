"""Turn what someone said into one of the 10 commands, or refuse.

Two levels, cheapest first:
  1. word rules: the things named in the sentence match exactly one command
  2. language model: picks a command number, or 0 for "can't do that"
When the word rules are in any doubt, the sentence goes to the language model.
"""
import re
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from tasks import COMMANDS

LLM_MODEL = "Qwen/Qwen3.5-4B"

# The things in the scene, and the words people use for them.
THINGS = {
    "bowl": r"\bbowl\b",
    "plate": r"\bplate\b",
    "wine": r"\b(wine|bottle)\b",
    "cheese": r"\bcheese\b",
    "stove": r"\b(stove|burner|cooktop|hob|gas)\b",
    "cabinet": r"\b(cabinet|cupboard|dresser)\b",
    "rack": r"\b(rack|holder)\b",
    "front": r"\bfront\b",
    "top_drawer": r"\b(top|upper|uppermost|highest)\s+(\w+\s+)?drawer\b",
    "middle_drawer": r"\b(middle|center|centre|second)\s+(\w+\s+)?drawer\b",
    "bottom_drawer": r"\b(bottom|lower|lowest|last)\s+(\w+\s+)?drawer\b",
}
DRAWERS = {"top_drawer", "middle_drawer", "bottom_drawer"}

# Words that could change the meaning, so the word rules step aside.
DOUBT = r"\b(off|close|closed|shut|remove|out|away|from|not|dont|but|instead|then|after|before|without|under|behind|next|beside|left|right)\b"

# (command id, things that must be named, thing that must be named first, action word needed)
RULES = [
    (0, {"middle_drawer"}, None, r"\b(open|pull|slide)\b"),
    (1, {"bowl", "stove"}, "bowl", None),
    (2, {"wine", "cabinet"}, "wine", None),
    (3, {"top_drawer", "bowl"}, None, None),
    (4, {"bowl", "cabinet"}, "bowl", None),
    (5, {"plate", "stove", "front"}, "plate", None),
    (6, {"cheese", "bowl"}, "cheese", None),
    (7, {"stove"}, None, r"\b(turn|switch|light|start|ignite|fire)\b"),
    (8, {"bowl", "plate"}, "bowl", None),
    (9, {"wine", "rack"}, "wine", None),
]


@dataclass
class Decision:
    command_id: int | None  # None means refuse
    level: str  # "rule" or "llm"
    ms: float


def clean(text):
    """Lowercase, keep only letters and single spaces, and join don't into dont."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower().replace("'", ""))).strip()


def named_things(text):
    """The scene things this sentence names. A drawer already implies the cabinet."""
    found = {name for name, pattern in THINGS.items() if re.search(pattern, text)}
    return found - {"cabinet"} if found & DRAWERS else found


def first_named(text, things):
    """Which of these things appears first in the sentence."""
    return min(things, key=lambda name: re.search(THINGS[name], text).start())


def match_rules(text):
    """Level 1: return a command id, or None if the rules are in any doubt."""
    text = clean(text)
    if re.search(DOUBT, text):
        return None
    found = named_things(text)
    for command, things, first, action in RULES:
        if found == things and (first is None or first_named(text, things) == first):
            if action is None or re.search(action, text):
                return command
    return None


class LanguageModel:
    """Level 2 model: picks a command number from a short list."""

    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(LLM_MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(LLM_MODEL, dtype=torch.float16).to(device).eval()
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(COMMANDS))
        self.system = (
            "You control a robot arm. It can only do these commands:\n"
            f"{numbered}\n"
            "The user speaks one request. Reply with only the number of the command that means the "
            "same thing. If it asks for a different action, object, place or drawer, or anything "
            "else, reply 0."
        )

    @torch.inference_mode()
    def choose(self, text):
        """Return a command id, or None if the model says it can't be done."""
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": text}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        reply = self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        digits = re.search(r"\d+", reply)
        number = int(digits.group()) if digits else 0
        return number - 1 if 1 <= number <= len(COMMANDS) else None


class Router:
    def __init__(self, device="cuda", use_rules=True):
        self.use_rules = use_rules
        self.llm = LanguageModel(device)

    def route(self, text):
        start = time.perf_counter()
        if self.use_rules:
            command = match_rules(text)
            if command is not None:
                return Decision(command, "rule", (time.perf_counter() - start) * 1000)
        command = self.llm.choose(text)
        torch.cuda.synchronize()
        if self.use_rules and command is not None:
            # Refuse if the model picked a command that doesn't include something the person named.
            allowed = next(things for c, things, _, _ in RULES if c == command)
            if named_things(clean(text)) - allowed:
                command = None
        return Decision(command, "llm", (time.perf_counter() - start) * 1000)
