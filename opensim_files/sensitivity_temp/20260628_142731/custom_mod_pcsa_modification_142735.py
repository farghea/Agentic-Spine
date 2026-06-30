import re

def modify_model(input_osim: str, value: float, output_osim: str) -> None:
    with open(input_osim, "r", encoding="utf-8") as f:
        content = f.read()

    muscle_block_pattern = re.compile(
        r'(<(?P<tag>Thelen2003Muscle|Millard2012EquilibriumMuscle|Schutte1993Muscle_Deprecated|Delp1990Muscle|RigidTendonMuscle)\b[^>]*>)(?P<body>.*?)(</(?P=tag)>)',
        re.DOTALL
    )

    max_iso_pattern = re.compile(
        r'(<max_isometric_force>\s*)([\d.eE+\-]+)(\s*</max_isometric_force>)',
        re.DOTALL
    )

    def scale_max_isometric_force(block_match):
        start_tag = block_match.group(1)
        body = block_match.group("body")
        end_tag = block_match.group(4)

        def repl_force(m):
            new_val = float(m.group(2)) * value
            return f"{m.group(1)}{new_val:.10f}{m.group(3)}"

        new_body, count = max_iso_pattern.subn(repl_force, body, count=1)
        return start_tag + new_body + end_tag

    modified = muscle_block_pattern.sub(scale_max_isometric_force, content)

    with open(output_osim, "w", encoding="utf-8") as f:
        f.write(modified)