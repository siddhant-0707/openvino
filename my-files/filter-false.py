with open('audiogen-1.5b-unsup.txt', 'r') as file:
    lines = file.readlines()

with open('audiogen-1.5b-unsup-fil.txt', 'w') as file:
    written_lines = set()
    for line in lines:
        if line.strip().endswith('False') and line not in written_lines:
            file.write(line)
            written_lines.add(line)
