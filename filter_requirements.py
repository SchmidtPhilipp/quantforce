import re


def is_pypi_package(line):
    # Skip empty lines and comments
    if not line.strip() or line.strip().startswith("#"):
        return False

    # Skip lines with @ file:// or @ file:/
    if "@ file:" in line:
        return False

    # Skip lines with @ /Users/ or @ /home/
    if "@ /" in line:
        return False

    return True


def main():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    filtered_lines = [line for line in lines if is_pypi_package(line)]

    with open("filtered_requirements.txt", "w") as f:
        f.writelines(filtered_lines)

    print(f"Filtered {len(lines) - len(filtered_lines)} problematic lines")
    print(
        f"Saved {len(filtered_lines)} clean package specifications to filtered_requirements.txt"
    )


if __name__ == "__main__":
    main()
