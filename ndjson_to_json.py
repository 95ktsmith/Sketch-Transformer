#!/usr/bin/env python3
"""
Converts files in ndjson format to a list of dict objects in standard
JSON format.

Usage: ./ndjson_to_json <file>.ndjson

Output: <file of same name>.json
"""

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: ./ndjson_to_json <file>.ndjson")
        sys.exit()

    lines = []
    with open(sys.argv[1], "r") as f:
        line = f.readline()
        while line:
            lines.append(json.loads(line))
            line = f.readline()

    with open("{}.json".format(sys.argv[1][:-7]), "w") as f:
        f.write(json.dumps(lines))
