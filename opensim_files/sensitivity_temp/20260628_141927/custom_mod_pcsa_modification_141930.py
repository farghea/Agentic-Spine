import re

def modify_model(input_osim: str, value: float, output_osim: str) -> None:
    with open(input_osim, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'(<max_isometric_force>\s*)([\d.eE+\-]+)(\s*</max_isometric_force>)'

    def repl(match):
        original = float(match.group(2))
        scaled = original * value
        return f"{match.group(1)}{scaled:.10f}{match.group(3)}"

    modified = re.sub(pattern, repl, content)

    with open(output_osim, 'w', encoding='utf-8') as f:
        f.write(modified)