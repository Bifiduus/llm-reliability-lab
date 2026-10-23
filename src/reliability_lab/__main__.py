import argparse


def main():
    parser = argparse.ArgumentParser(
        description="LLM Reliability Lab"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="LLM Reliability Lab 0.1.0"
    )

    args = parser.parse_args()

    print("LLM Reliability Lab is ready!")


if __name__ == "__main__":
    main()