from pathlib import Path
from scipy.io import loadmat
import numpy as np

# ============================================================
# STAGE 0A — INSPECT THE GUIDANCE DATASET STRUCTURE
# ============================================================
# Purpose:
#   Inspect the original MATLAB guidance dataset without
#   modifying, filtering, or analysing the data.
#
# Dataset:
#   Performance assessment of no-fee GNSS augmentation
#   systems for tractor guidance (Gomez-Gil et al., 2026)
# ============================================================


# ------------------------------------------------------------
# 1. File path
# ------------------------------------------------------------

MAT_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "source"
    / "2_Guidance_tests"
    / "2A_Guidance_tests_data.mat"
)

print("=" * 80)
print("GUIDANCE DATASET INSPECTION")
print("=" * 80)

print("\nMAT file:")
print(MAT_FILE)

if not MAT_FILE.exists():
    raise FileNotFoundError(
        "\nThe MATLAB file was not found.\n"
        f"Path checked:\n{MAT_FILE}\n"
    )

print("\nFile found successfully.")
print(f"File size: {MAT_FILE.stat().st_size / (1024**2):.2f} MB")


# ------------------------------------------------------------
# 2. Load MATLAB file
# ------------------------------------------------------------

print("\nLoading MATLAB file...")

data = loadmat(
    MAT_FILE,
    struct_as_record=False,
    squeeze_me=True
)

print("MATLAB file loaded successfully.")


# ------------------------------------------------------------
# 3. Remove MATLAB metadata variables
# ------------------------------------------------------------

variables = {
    key: value
    for key, value in data.items()
    if not key.startswith("__")
}


# ------------------------------------------------------------
# 4. Show top-level variables
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TOP-LEVEL VARIABLES")
print("=" * 80)

print(f"\nNumber of user variables: {len(variables)}")

for i, (key, value) in enumerate(variables.items(), start=1):

    print("\n" + "-" * 80)
    print(f"[{i}] Variable name : {key}")
    print(f"    Python type   : {type(value).__name__}")

    if hasattr(value, "shape"):
        print(f"    Shape         : {value.shape}")

    if hasattr(value, "dtype"):
        print(f"    Data type     : {value.dtype}")

    if hasattr(value, "_fieldnames"):
        print(f"    MATLAB fields : {value._fieldnames}")


# ------------------------------------------------------------
# 5. Recursive structure inspector
# ------------------------------------------------------------

def inspect_object(obj, name="root", depth=0, max_depth=5):
    """
    Recursively inspect MATLAB structures loaded with scipy.io.loadmat.

    This function only reports structure and metadata.
    It does not modify the dataset.
    """

    indent = "    " * depth

    print(f"{indent}└── {name}")
    print(f"{indent}    type: {type(obj).__name__}")

    if hasattr(obj, "shape"):
        print(f"{indent}    shape: {obj.shape}")

    if isinstance(obj, np.ndarray):
        print(f"{indent}    dtype: {obj.dtype}")

    if depth >= max_depth:
        print(f"{indent}    [maximum inspection depth reached]")
        return

    # MATLAB structure
    if hasattr(obj, "_fieldnames"):

        fields = obj._fieldnames

        print(f"{indent}    fields: {fields}")

        for field in fields:

            try:
                child = getattr(obj, field)

                inspect_object(
                    child,
                    name=field,
                    depth=depth + 1,
                    max_depth=max_depth
                )

            except Exception as exc:

                print(
                    f"{indent}        Could not inspect "
                    f"{field}: {exc}"
                )

        return

    # NumPy structured array
    if isinstance(obj, np.ndarray):

        if obj.dtype.names is not None:

            print(
                f"{indent}    structured fields: "
                f"{obj.dtype.names}"
            )

            for field in obj.dtype.names:

                try:
                    inspect_object(
                        obj[field],
                        name=field,
                        depth=depth + 1,
                        max_depth=max_depth
                    )

                except Exception as exc:

                    print(
                        f"{indent}        Could not inspect "
                        f"{field}: {exc}"
                    )

            return

        # Object arrays may contain MATLAB structs
        if obj.dtype == object and obj.size > 0:

            print(
                f"{indent}    object-array elements: "
                f"{obj.size}"
            )

            # Inspect only the first few elements so that
            # the terminal output does not become enormous.
            number_to_inspect = min(3, obj.size)

            flat = obj.ravel()

            for index in range(number_to_inspect):

                inspect_object(
                    flat[index],
                    name=f"element_{index}",
                    depth=depth + 1,
                    max_depth=max_depth
                )

            if obj.size > number_to_inspect:

                print(
                    f"{indent}    ... "
                    f"{obj.size - number_to_inspect} "
                    f"additional elements not expanded"
                )

            return


# ------------------------------------------------------------
# 6. Detailed inspection
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DETAILED MATLAB STRUCTURE")
print("=" * 80)

for key, value in variables.items():

    print("\n" + "-" * 80)

    inspect_object(
        value,
        name=key,
        depth=0,
        max_depth=5
    )


# ------------------------------------------------------------
# 7. Look for configuration-related names
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("POTENTIAL CONFIGURATION VARIABLES")
print("=" * 80)

configuration_keywords = [
    "config",
    "configuration",
    "receiver",
    "guidance",
    "reference",
    "trajectory",
    "position",
    "north",
    "east",
    "time",
    "error"
]

found = False

for key in variables:

    key_lower = key.lower()

    if any(
        keyword in key_lower
        for keyword in configuration_keywords
    ):

        print(f"  • {key}")
        found = True

if not found:
    print(
        "No obvious configuration-related variable names "
        "were found at the top level."
    )


# ------------------------------------------------------------
# 8. Finish
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

print(
    "\nNo calculations, filtering, transformations, "
    "or modifications were performed."
)

print(
    "\nNext step: interpret the MATLAB structure and identify "
    "the exact variables corresponding to configurations C1–C14, "
    "trajectory coordinates, reference data, passes, and time."
)