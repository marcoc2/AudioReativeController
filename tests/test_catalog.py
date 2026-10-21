"""The catalog of Generators and FX: every layer source and shader in it, kinds read from the code."""
from core import catalog as c
from core.video.layers import LAYER_SOURCES


def test_every_layer_source_is_listed():
    missing = [s for s in LAYER_SOURCES if s not in c.NATIVE and s not in c.NOT_LISTED]
    assert missing == [], f"add these to core/catalog.py NATIVE: {missing}"


def test_every_shader_is_listed_with_a_category():
    keys = {e.key for e in c.entries()}
    for path in c.shader_paths():
        assert f"shadertoy:{path.stem}" in keys
        assert path.stem in c.SHADER_CATEGORIES, f"give {path.stem} a category in core/catalog.py"


def test_categories_belong_to_their_kind():
    for e in c.entries():
        cats = c.GENERATOR_CATEGORIES if e.kind == c.GENERATOR else c.FX_CATEGORIES
        assert e.category in cats, (e.key, e.kind, e.category)


def test_kinds_come_from_the_code():
    assert c.by_key("julia").kind == c.GENERATOR
    assert c.by_key("grain").kind == c.FX
    assert c.by_key("flame").takes_input                         # paint: under
    assert c.by_key("shadertoy:droste_arc").kind == c.FX          # reads frame
    assert c.by_key("shadertoy:nebulosa_arc").kind == c.GENERATOR
    assert c.by_key("shadertoy:kifs_acorde_arc").takes_input      # u_video
    assert c.by_key("cast").needs == ("masks",)
    assert c.by_key("shadertoy:cenario_arc").needs == ("masks",)


def test_params_from_the_docstring_example_and_the_code():
    grain = {p.name: p for p in c.by_key("grain").params}
    assert grain["amount"].default == 0.06 and "strength" in grain["amount"].about
    assert grain["amount"].knob and "bars" not in grain
    feedback = {p.name: p for p in c.by_key("feedback").params}        # no example: read from the code
    assert feedback["decay"].default == 0.85
    toy = {p.name for p in c.by_key("shadertoy:droste_arc").params}
    assert toy and all(n.startswith("u_") for n in toy)


def test_tree_and_search():
    cats = [cat for cat, _ in c.tree(c.GENERATOR)]
    assert cats[0] == "Fractals"
    labels = [e.label for _, es in c.tree(c.GENERATOR) for e in es]
    assert "kaleidoscope_arc" in labels and "kaleidoscope" not in labels    # the _arc copy stands for it
    found = [e.label for _, es in c.tree(c.FX, "grain") for e in es]
    assert found == ["grain"]


def test_a_layer_knows_its_entry():
    assert c.of_layer({"source": "shadertoy", "file": "shaders/shadertoy/droste_arc.glsl"}).kind == c.FX
    assert c.of_layer({"source": "footage", "file": "x.mp4"}) is None
    assert c.by_key("shadertoy:bloom_arc").spec() == {"source": "shadertoy", "file": "shaders/shadertoy/bloom_arc"}
