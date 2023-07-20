import subprocess

def generate_mermaid_svg(mermaid_code, output_filename):
    # Write the Mermaid code to a temporary file
    with open("temp.mmd", "w") as f:
        f.write(mermaid_code)
    result = None
    # Generate the SVG using mmdc
    try:
        result = subprocess.run(['YOUR_MMDC_PATH', "-i", "temp.mmd", "-o", output_filename], check=True)
    except subprocess.CalledProcessError:
        # An error occurred while running mmdc
        print(result.stdout, "gadbad hai.......")
        # print(result.stderr)
        return None
    
    # Read the generated SVG file
    with open(output_filename, "r") as f:
        svg = f.read()

    return svg

if __name__ == "__main__":
    import sys
    mermaid_code = sys.argv[1]
    output_filename = sys.argv[2]
    svg = generate_mermaid_svg(mermaid_code, output_filename)
    print(svg)
