"""Golden test: the resolved view must reproduce the legacy registry exactly
(semantically), and every entry must be reachable by the C engine's
strstr-based parser."""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Hermetic: the developer's real per-user state (enabled models, default)
# must not influence these tests in either direction.
_TMP_CONFIG = tempfile.mkdtemp(prefix="flashchat-test-config.")
os.environ["FLASHCHAT_CONFIG_DIR"] = _TMP_CONFIG

from modelmgr import resolved
from modelmgr.registry import Registry, resolved_id

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "model_configs_legacy.json")


def normalize(value):
    """Make int/float JSON differences (10000000 vs 10000000.0) compare equal."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


class TestGoldenEquivalence(unittest.TestCase):
    """Rendering the shipped manifests must reproduce the legacy registry the
    C engine was built against, field for field."""

    @classmethod
    def setUpClass(cls):
        os.environ["FLASHCHAT_CONFIG_DIR"] = _TMP_CONFIG
        with open(GOLDEN) as f:
            cls.golden = json.load(f)
        cls.view = resolved.render(Registry.load(), include_all=True)

    def test_same_model_ids(self):
        self.assertEqual(set(self.view["models"]), set(self.golden["models"]))

    def test_entries_semantically_identical(self):
        for mid, golden_entry in self.golden["models"].items():
            rendered = self.view["models"][mid]
            self.assertEqual(
                normalize(golden_entry), normalize(rendered),
                f"entry '{mid}' diverges from legacy registry",
            )

    def test_default_model_preserved(self):
        self.assertEqual(self.view["default_model"], self.golden["default_model"])

    def test_server_defaults_preserved(self):
        self.assertEqual(self.view["server_defaults"], self.golden["server_defaults"])


class CParserEmulator:
    """Re-implements load_model_config()'s lookup discipline from
    metal_infer/model_config.h: strstr for '"<id>"', then scope key lookups
    to that brace-balanced object."""

    def __init__(self, text: str):
        self.text = text

    def entry_span(self, model_id: str):
        needle = f'"{model_id}"'
        start = self.text.find(needle)
        if start == -1:
            return None
        obj_start = self.text.find("{", start + len(needle))
        depth = 0
        in_string = False
        escaped = False
        for i in range(obj_start, len(self.text)):
            c = self.text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return (obj_start, i + 1)
        return None

    def scoped_value(self, model_id: str, key: str):
        span = self.entry_span(model_id)
        if span is None:
            return None
        body = self.text[span[0]:span[1]]
        pos = body.find(f'"{key}"')
        if pos == -1:
            return None
        colon = body.index(":", pos)
        rest = body[colon + 1:].lstrip()
        if rest.startswith('"'):
            return rest[1:rest.index('"', 1)]
        end = min((rest.index(c) for c in ",}\n" if c in rest), default=len(rest))
        return rest[:end].strip()


class TestStrstrScoping(unittest.TestCase):
    """Each id's first occurrence in the serialized file must be its own
    model key, and scoped lookups must return that entry's values."""

    @classmethod
    def setUpClass(cls):
        os.environ["FLASHCHAT_CONFIG_DIR"] = _TMP_CONFIG
        cls.registry = Registry.load()
        cls.view = resolved.render(cls.registry, include_all=True)
        cls.text = json.dumps(cls.view, indent=2)
        cls.parser = CParserEmulator(cls.text)

    def test_models_section_precedes_default_model(self):
        self.assertLess(self.text.find('"models"'), self.text.find('"default_model"'))

    def test_each_id_scopes_to_own_entry(self):
        for m in self.registry.manifests.values():
            for vname, variant in m.variants.items():
                rid = resolved_id(m, vname)
                bits = self.parser.scoped_value(rid, "bits")
                self.assertEqual(int(bits), variant.bits, f"{rid}: bits mis-scoped")
                name = self.parser.scoped_value(rid, "name")
                self.assertEqual(name, f"{m.name} ({vname})", f"{rid}: name mis-scoped")

    def test_thinking_capable_not_inherited_across_entries(self):
        # The regression the C parser comments warn about: a 35B entry must
        # not pick up Qwen3-Next's thinking_capable=false.
        for rid in ("Qwen-Qwen36-35B-A3B", "Qwen-Qwen36-35B-A3B-8bit"):
            self.assertIsNone(self.parser.scoped_value(rid, "thinking_capable"), rid)
        for rid in ("Qwen-Qwen3-Next-80B-A3B-Instruct", "Qwen-Qwen3-Next-80B-A3B-Instruct-8bit"):
            self.assertEqual(self.parser.scoped_value(rid, "thinking_capable"), "false", rid)


class TestLegacyLookup(unittest.TestCase):
    def test_every_legacy_id_maps_back(self):
        os.environ["FLASHCHAT_CONFIG_DIR"] = _TMP_CONFIG
        registry = Registry.load()
        with open(GOLDEN) as f:
            golden = json.load(f)
        for legacy_id, entry in golden["models"].items():
            hit = registry.lookup_legacy(legacy_id)
            self.assertIsNotNone(hit, f"legacy id '{legacy_id}' unmapped")
            manifest, vname = hit
            self.assertEqual(manifest.hf_repo, entry["hf_repo"])
            self.assertEqual(manifest.variant(vname).bits, entry["quantization"]["bits"])
            self.assertEqual(resolved_id(manifest, vname), legacy_id)


if __name__ == "__main__":
    unittest.main()
